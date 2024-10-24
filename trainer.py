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
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

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

        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        writer = SummaryWriter(log_dir=self.args.model_dir)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

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
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.args.warmup_steps, t_total)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Total optimization steps = {t_total}")

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)

                # Unpack batch depending on contrastive learning flag
                if self.args.use_contrastive_learning:
                    inputs = {
                        "input_ids": batch[0],  # Anchor input
                        "attention_mask": batch[3],
                        "token_type_ids": batch[6],
                        "positive_input_ids": batch[1],  # Positive input
                        "positive_attention_mask": batch[4],
                        "negative_input_ids": batch[2],  # Negative input
                        "negative_attention_mask": batch[5],
                        "intent_label_ids": batch[9],  # Intent labels
                        "slot_labels_ids": batch[10],  # Slot labels
                        "positive_token_type_ids": batch[7],
                        "negative_token_type_ids": batch[8],
                    }
                else:
                    inputs = {
                        "input_ids": batch[0],  # Regular input
                        "attention_mask": batch[3],
                        "token_type_ids": batch[6],
                        "intent_label_ids": batch[9],
                        "slot_labels_ids": batch[10],
                    }

                outputs = self.model(**inputs)
                loss = outputs[0]

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
                        writer.add_scalar("Loss/validation", results["loss"], global_step)

                        early_stopping(results[self.args.tuning_metric], self.model, self.args)
                        if early_stopping.early_stop:
                            logger.info("Early stopping")
                            break

                if 0 < self.args.max_steps <= global_step:
                    epoch_iterator.close()
                    break

            if early_stopping.early_stop:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        dataset = self.dev_dataset if mode == "dev" else self.test_dataset
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        logger.info(f"***** Running evaluation on {mode} dataset *****")
        logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Batch size = {self.args.eval_batch_size}")

        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds, slot_preds = None, None
        out_intent_label_ids, out_slot_labels_ids = None, None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[3],
                    "token_type_ids": batch[6],
                    "intent_label_ids": batch[9],
                    "slot_labels_ids": batch[10],
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                # Collect predictions
                if intent_preds is None:
                    intent_preds = intent_logits.detach().cpu().numpy()
                    out_intent_label_ids = inputs["intent_label_ids"].detach().cpu().numpy()
                else:
                    intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                    out_intent_label_ids = np.append(out_intent_label_ids, inputs["intent_label_ids"].detach().cpu().numpy(), axis=0)

                if slot_preds is None:
                    slot_preds = np.array(self.model.crf.decode(slot_logits)) if self.args.use_crf else slot_logits.detach().cpu().numpy()
                    out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

        eval_loss /= nb_eval_steps
        results = {"loss": eval_loss}
        intent_preds = np.argmax(intent_preds, axis=1)

        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds, out_slot_labels_ids)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key, value in results.items():
            logger.info(f"  {key} = {value}")

        return results
