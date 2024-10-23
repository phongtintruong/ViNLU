import copy
import json
import logging
import os
import random

import torch
from torch.utils.data import TensorDataset
from utils import get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)

        self.input_text_file = "seq.in"
        self.intent_label_file = "label"
        self.slot_labels_file = "seq.out"

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, intents, slots, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. intent
            intent_label = (
                self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            )
            # 3. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(
                    self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK")
                )

            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.token_level, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(
            texts=self._read_file(os.path.join(data_path, self.input_text_file)),
            intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
            slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
            set_type=mode,
        )


processors = {"syllable-level": JointProcessor, "word-level": JointProcessor}


def convert_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
    pad_token_label_id=-100,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
            len(slot_labels_ids), max_seq_len
        )

        intent_label_id = int(example.intent_label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                intent_label_id=intent_label_id,
                slot_labels_ids=slot_labels_ids,
            )
        )

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.token_level](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode, args.token_level, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(
            examples, args.max_seq_len, tokenizer, pad_token_label_id=pad_token_label_id
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_intent_label_ids, all_slot_labels_ids
    )
    return dataset


class TripletInputExample(object):
    """
    A single training/test example for triplet-based contrastive learning.

    Args:
        guid: Unique id for the example.
        anchor_words: list. The words of the anchor sentence.
        positive_words: list. The words of the positive sentence (same intent).
        negative_words: list. The words of the negative sentence (different intent).
        intent_label: (Optional) string. The intent label of the anchor example.
        slot_labels: (Optional) list. The slot labels of the anchor example.
    """

    def __init__(self, guid, anchor_words, positive_words, negative_words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.anchor_words = anchor_words
        self.positive_words = positive_words
        self.negative_words = negative_words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TripletInputFeatures(object):
    """
    A single set of features for triplet-based contrastive learning.

    Args:
        anchor_input_ids: Input IDs for the anchor sentence.
        positive_input_ids: Input IDs for the positive sentence (same intent).
        negative_input_ids: Input IDs for the negative sentence (different intent).
        anchor_attention_mask: Attention mask for the anchor sentence.
        positive_attention_mask: Attention mask for the positive sentence.
        negative_attention_mask: Attention mask for the negative sentence.
        anchor_token_type_ids: Token type IDs for the anchor sentence.
        positive_token_type_ids: Token type IDs for the positive sentence.
        negative_token_type_ids: Token type IDs for the negative sentence.
        intent_label_id: The intent label ID for the anchor sentence.
        slot_labels_ids: The slot labels IDs for the anchor sentence.
    """

    def __init__(self, anchor_input_ids, positive_input_ids, negative_input_ids,
                 anchor_attention_mask, positive_attention_mask, negative_attention_mask,
                 anchor_token_type_ids, positive_token_type_ids, negative_token_type_ids,
                 intent_label_id, slot_labels_ids):
        self.anchor_input_ids = anchor_input_ids
        self.positive_input_ids = positive_input_ids
        self.negative_input_ids = negative_input_ids
        self.anchor_attention_mask = anchor_attention_mask
        self.positive_attention_mask = positive_attention_mask
        self.negative_attention_mask = negative_attention_mask
        self.anchor_token_type_ids = anchor_token_type_ids
        self.positive_token_type_ids = positive_token_type_ids
        self.negative_token_type_ids = negative_token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TripletProcessor(object):
    """
    Processor for creating triplet examples from the dataset for contrastive learning.

    Args:
        args: Arguments such as data directory, token level, etc.
    """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)

        self.input_text_file = "seq.in"
        self.intent_label_file = "label"
        self.slot_labels_file = "seq.out"

    @classmethod
    def _read_file(cls, input_file):
        """Reads a file line by line."""
        with open(input_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]

    def _create_triplet_examples(self, texts, intents, slots, set_type):
        """
        Creates triplet examples for contrastive learning (anchor, positive, negative).

        Args:
            texts: List of sentences.
            intents: List of intents corresponding to the sentences.
            slots: List of slot labels for each sentence.
            set_type: Specifies if it's 'train', 'dev', or 'test'.
        """
        examples = []
        intent_to_sentences = {}

        # Group sentences by intent
        for i, intent in enumerate(intents):
            if intent not in intent_to_sentences:
                intent_to_sentences[intent] = []
            intent_to_sentences[intent].append((texts[i], slots[i]))

        # Generate triplets (anchor, positive, negative)
        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            guid = f"{set_type}-{i}"
            words = text.split()

            # Positive sample (same intent)
            positive_text, positive_slot = random.choice(intent_to_sentences[intent])
            positive_words = positive_text.split()

            # Negative sample (different intent)
            negative_intent = random.choice([k for k in intent_to_sentences.keys() if k != intent])
            negative_text, negative_slot = random.choice(intent_to_sentences[negative_intent])
            negative_words = negative_text.split()

            examples.append(TripletInputExample(
                guid=guid,
                anchor_words=words,
                positive_words=positive_words,
                negative_words=negative_words,
                intent_label=intent,
                slot_labels=slot
            ))

        return examples

    def get_triplet_examples(self, mode):
        """
        Retrieves triplet examples for the specified mode (train, dev, test).

        Args:
            mode: Specifies the dataset split ('train', 'dev', 'test').
        """
        data_path = os.path.join(self.args.data_dir, self.args.token_level, mode)
        logger.info(f"LOOKING AT {data_path}")
        return self._create_triplet_examples(
            texts=self._read_file(os.path.join(data_path, self.input_text_file)),
            intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
            slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
            set_type=mode,
        )


def convert_triplet_examples_to_features(
        examples, max_seq_len, tokenizer, pad_token_label_id=-100, cls_token_segment_id=0,
        pad_token_segment_id=0, sequence_a_segment_id=0, mask_padding_with_zero=True):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id
    features = []

    # Load intent labels for mapping intent strings to integer IDs
    intent_label_list = get_intent_labels(None)  # Assuming you load them globally or pass args

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info(f"Writing example {ex_index} of {len(examples)}")

        # Tokenize sentences
        anchor_input_ids, anchor_attention_mask = tokenize_sentence(example.anchor_words, tokenizer, max_seq_len)
        positive_input_ids, positive_attention_mask = tokenize_sentence(example.positive_words, tokenizer, max_seq_len)
        negative_input_ids, negative_attention_mask = tokenize_sentence(example.negative_words, tokenizer, max_seq_len)

        # Map intent label string to an integer ID
        try:
            intent_label_id = intent_label_list.index(example.intent_label)
        except ValueError:
            raise ValueError(f"Intent label '{example.intent_label}' not found in intent label list.")

        features.append(
            TripletInputFeatures(
                anchor_input_ids=anchor_input_ids,
                positive_input_ids=positive_input_ids,
                negative_input_ids=negative_input_ids,
                anchor_attention_mask=anchor_attention_mask,
                positive_attention_mask=positive_attention_mask,
                negative_attention_mask=negative_attention_mask,
                anchor_token_type_ids=[sequence_a_segment_id] * max_seq_len,
                positive_token_type_ids=[sequence_a_segment_id] * max_seq_len,
                negative_token_type_ids=[sequence_a_segment_id] * max_seq_len,
                intent_label_id=intent_label_id,
                slot_labels_ids=[pad_token_label_id] * max_seq_len
            )
        )

    return features


def tokenize_sentence(words, tokenizer, max_seq_len):
    """Tokenizes and processes a sentence."""
    tokens = []
    for word in words:
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)

    # Apply special tokens and padding
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[: (max_seq_len - 2)]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    attention_mask = [1] * len(input_ids)
    padding_length = max_seq_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length

    return input_ids, attention_mask


def load_and_cache_triplet_examples(args, tokenizer, mode):
    """
    Loads and caches triplet examples for contrastive learning.

    Args:
        args: Arguments containing data and model configuration.
        tokenizer: Pretrained tokenizer.
        mode: Dataset mode ('train', 'dev', 'test').
    """
    processor = TripletProcessor(args)

    cached_features_file = os.path.join(
        args.data_dir,
        f"cached_triplet_{mode}_{args.token_level}_{args.max_seq_len}"
    )

    if os.path.exists(cached_features_file):
        logger.info(f"Loading triplet features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        logger.info(f"Creating triplet features from dataset at {args.data_dir}")
        examples = processor.get_triplet_examples(mode)

        features = convert_triplet_examples_to_features(
            examples, args.max_seq_len, tokenizer, pad_token_label_id=args.ignore_index
        )
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset for triplet loss
    all_anchor_input_ids = torch.tensor([f.anchor_input_ids for f in features], dtype=torch.long)
    all_positive_input_ids = torch.tensor([f.positive_input_ids for f in features], dtype=torch.long)
    all_negative_input_ids = torch.tensor([f.negative_input_ids for f in features], dtype=torch.long)

    all_anchor_attention_mask = torch.tensor([f.anchor_attention_mask for f in features], dtype=torch.long)
    all_positive_attention_mask = torch.tensor([f.positive_attention_mask for f in features], dtype=torch.long)
    all_negative_attention_mask = torch.tensor([f.negative_attention_mask for f in features], dtype=torch.long)

    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_anchor_input_ids, all_positive_input_ids, all_negative_input_ids,
        all_anchor_attention_mask, all_positive_attention_mask, all_negative_attention_mask,
        all_intent_label_ids
    )
    return dataset