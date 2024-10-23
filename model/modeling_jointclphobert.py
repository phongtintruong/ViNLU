import torch
import torch.nn as nn
from torchcrf import CRF
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

from .module import IntentClassifier, SlotClassifier


class JointCLPhoBERT(RobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointCLPhoBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.roberta = RobertaModel(config)  # Load pretrained phobert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)

        self.slot_classifier = SlotClassifier(
            config.hidden_size,
            self.num_intent_labels,
            self.num_slot_labels,
            self.args.use_intent_context_concat,
            self.args.use_intent_context_attention,
            self.args.max_seq_len,
            self.args.attention_embedding_size,
            args.dropout_rate,
        )

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        # Add flag to determine whether to use contrastive learning
        self.use_contrastive_learning = args.use_contrastive_learning

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids=None, slot_labels_ids=None,
                anchor=None, positive=None, negative=None):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # 1. Intent Classification
        intent_logits = self.intent_classifier(pooled_output)

        # 2. Slot Classification
        if not self.args.use_attention_mask:
            tmp_attention_mask = None
        else:
            tmp_attention_mask = attention_mask

        if self.args.embedding_type == "hard":
            hard_intent_logits = torch.zeros(intent_logits.shape)
            for i, sample in enumerate(intent_logits):
                max_idx = torch.argmax(sample)
                hard_intent_logits[i][max_idx] = 1
            slot_logits = self.slot_classifier(sequence_output, hard_intent_logits, tmp_attention_mask)
        else:
            slot_logits = self.slot_classifier(sequence_output, intent_logits, tmp_attention_mask)

        total_loss = 0
        # 3. Intent Loss
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
                )
            total_loss += self.args.intent_loss_weight * intent_loss

        # 4. Slot Loss
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction="mean")
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_weight * slot_loss

        # 5. Add Contrastive Loss (Triplet Loss)
        if self.use_contrastive_learning and anchor is not None and positive is not None and negative is not None:
            contrastive_loss = self.compute_contrastive_loss(anchor, positive, negative)
            total_loss += self.args.contrastive_loss_weight * contrastive_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits

    def compute_contrastive_loss(self, anchor, positive, negative, margin=1.0):
        """
        Compute contrastive loss using triplet loss between anchor, positive, and negative embeddings.
        """
        # Calculate Euclidean distances
        positive_distance = nn.functional.pairwise_distance(anchor, positive)
        negative_distance = nn.functional.pairwise_distance(anchor, negative)

        # Triplet loss
        triplet_loss = nn.functional.relu(positive_distance - negative_distance + margin)
        return triplet_loss.mean()
