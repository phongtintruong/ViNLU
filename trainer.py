import logging
import os
import numpy as np
import torch
from early_stopping import EarlyStopping
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        self.pad_token_label_id = args.ignore_index  # Ignore index for loss padding

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

        # Load model (pretrained or from scratch)
        if args.pretrained:
            self.model = self.model_class.from_pretrained(
                args.pretrained_path,
                args=args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst,
            )
        else:
            self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.token_level)
            self.model = self.model_class.from_pretrained(
                args.model_name_or_path,
                config=self.config,
                args=args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst,
            )

        # Set device (GPU/CPU)
        torch.cuda.set_device(args.gpu_id)
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        writer = SummaryWriter(log_dir=self.args.model_dir)

        # Calculate total training steps
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and scheduler
        optimizer_grouped_parameters = self._get_optimizer_params()
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        logger.info("***** Running training *****")
        global_step, tr_loss = 0, 0.0
        self.model.zero_grad()
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)

                # Prepare inputs based on whether contrastive learning is used
                if self.args.use_contrastive_learning:
                    inputs = self._prepare_contrastive_inputs(batch)
                else:
                    inputs = self._prepare_standard_inputs(batch)

                # Forward pass and compute loss
                outputs = self.model(**inputs)
                loss = outputs[0]  # Weighted total loss

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if global_step % self.args.logging_steps == 0:
                        results = self.evaluate("dev")
                        self._log_metrics(writer, results, global_step)
                        early_stopping(results[self.args.tuning_metric], self.model, self.args)
                        if early_stopping.early_stop:
                            logger.info("Early stopping")
                            break

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step or early_stopping.early_stop:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def _prepare_standard_inputs(self, batch):
        """Prepare inputs for standard joint learning."""
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "intent_label_ids": batch[3],
            "slot_labels_ids": batch[4],
        }
        if self.args.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2]
        return inputs

    def _prepare_contrastive_inputs(self, batch):
        """Prepare inputs for contrastive learning."""
        inputs = self._prepare_standard_inputs(batch)
        inputs.update({
            "anchor": batch[5],
            "positive": batch[6],
            "negative": batch[7],
        })
        return inputs

    def _get_optimizer_params(self):
        """Get optimizer parameters with correct weight decay."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def _log_metrics(self, writer, results, global_step):
        """Log metrics to TensorBoard."""
        writer.add_scalar("Loss/validation", results["loss"], global_step)
        writer.add_scalar("Intent Accuracy/validation", results["intent_acc"], global_step)
        writer.add_scalar("Slot F1/validation", results["slot_f1"], global_step)
        writer.add_scalar("Semantic Frame Accuracy", results["semantic_frame_acc"], global_step)

    def evaluate(self, mode):
        """Evaluation loop for development or test datasets."""
        dataset = self.dev_dataset if mode == "dev" else self.test_dataset
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        logger.info(f"***** Running evaluation on {mode} dataset *****")
        eval_loss, nb_eval_steps = 0.0, 0
        intent_preds, slot_preds, out_intent_label_ids, out_slot_labels_ids = None, None, None, None

        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = self._prepare_standard_inputs(batch)
                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

                # Collect predictions
                intent_preds, out_intent_label_ids = self._append_preds(intent_preds, intent_logits, out_intent_label_ids, inputs["intent_label_ids"])
                slot_preds, out_slot_labels_ids = self._append_preds(slot_preds, slot_logits, out_slot_labels_ids, inputs["slot_labels_ids"])

        eval_loss /= nb_eval_steps
        results = {"loss": eval_loss}
        results.update(compute_metrics(intent_preds, out_intent_label_ids, slot_preds, out_slot_labels_ids))

        logger.info("***** Eval results *****")
        for key, value in results.items():
            logger.info(f"{key} = {value}")

        return results

    def _append_preds(self, preds, logits, out_labels, labels):
        """Helper to append predictions and labels."""
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_labels = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_labels = np.append(out_labels, labels.detach().cpu().numpy(), axis=0)
        return preds, out_labels
