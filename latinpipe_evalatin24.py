#!/usr/bin/env python3
#
# This file is part of LatinPipe EvaLatin 24
# <https://github.com/ufal/evalatin2024-latinpipe>.
#
# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import argparse
import collections
import datetime
import difflib
import io
import json
import os
import pickle
import re
from typing import Self
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch
import transformers
import ufal.chu_liu_edmonds

import latinpipe_evalatin24_eval

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--deprel", default="full", choices=["full", "universal"], type=str, help="Deprel kind.")
parser.add_argument("--dev", default=[], nargs="+", type=str, help="Dev CoNLL-U files.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
parser.add_argument("--embed_tags", default="", type=str, help="Tags to embed on input.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--epochs_frozen", default=0, type=int, help="Number of epochs with frozen transformer.")
parser.add_argument("--exp", default=None, type=str, help="Experiment name.")
parser.add_argument("--label_smoothing", default=0.03, type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
parser.add_argument("--learning_rate_decay", default="cos", choices=["none", "cos"], type=str, help="Learning rate decay.")
parser.add_argument("--learning_rate_warmup", default=2_000, type=int, help="Number of warmup steps.")
parser.add_argument("--load", default=[], type=str, nargs="*", help="Path to load models from.")
parser.add_argument("--max_train_sentence_len", default=150, type=int, help="Max sentence subwords in training.")
parser.add_argument("--optimizer", default="adam", choices=["adam", "adafactor"], type=str, help="Optimizer.")
parser.add_argument("--parse", default=1, type=int, help="Parse.")
parser.add_argument("--parse_attention_dim", default=512, type=int, help="Parse attention dimension.")
parser.add_argument("--rnn_dim", default=512, type=int, help="RNN layers size.")
parser.add_argument("--rnn_layers", default=2, type=int, help="RNN layers.")
parser.add_argument("--rnn_type", default="LSTMTorch", choices=["LSTM", "GRU", "LSTMTorch", "GRUTorch"], help="RNN type.")
parser.add_argument("--save_checkpoint", default=False, action="store_true", help="Save checkpoint.")
parser.add_argument("--seed", default=42, type=int, help="Initial random seed.")
parser.add_argument("--steps_per_epoch", default=1_000, type=int, help="Steps per epoch.")
parser.add_argument("--single_root", default=1, type=int, help="Single root allowed only.")
parser.add_argument("--subword_combination", default="first", choices=["first", "last", "sum", "concat"], type=str, help="Subword combination.")
parser.add_argument("--tags", default="UPOS,LEMMAS,FEATS", type=str, help="Tags to predict.")
parser.add_argument("--task_hidden_layer", default=2_048, type=int, help="Task hidden layer size.")
parser.add_argument("--test", default=[], nargs="+", type=str, help="Test CoNLL-U files.")
parser.add_argument("--train", default=[], nargs="+", type=str, help="Train CoNLL-U files.")
parser.add_argument("--train_sampling_exponent", default=0.5, type=float, help="Train sampling exponent.")
parser.add_argument("--transformers", nargs="+", type=str, help="Transformers models to use.")
parser.add_argument("--treebank_ids", default=False, action="store_true", help="Include treebank IDs on input.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verbose", default=2, type=int, help="Verbosity")
parser.add_argument("--wandb", default=False, action="store_true", help="Log in WandB.")
parser.add_argument("--word_masking", default=None, type=float, help="Word masking")


class UDDataset:
    FORMS, LEMMAS, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, FACTORS = range(10)
    FACTORS_MAP = {"FORMS": FORMS, "LEMMAS": LEMMAS, "UPOS": UPOS, "XPOS": XPOS, "FEATS": FEATS,
                   "HEAD": HEAD, "DEPREL": DEPREL, "DEPS": DEPS, "MISC": MISC}
    RE_EXTRAS = re.compile(r"^#|^\d+-|^\d+\.")

    class Factor:
        def __init__(self, train_factor: Self = None):
            self.words_map = train_factor.words_map if train_factor else {"<unk>": 0}
            self.words = train_factor.words if train_factor else ["<unk>"]
            self.word_ids = []
            self.strings = []

    def __init__(self, path: str, args: argparse.Namespace, treebank_id: int|None = None, train_dataset: Self = None, text: str|None = None):
        self.path = path

        # Create factors and other variables
        self.factors = []
        for f in range(self.FACTORS):
            self.factors.append(self.Factor(train_dataset.factors[f] if train_dataset is not None else None))
        self._extras = []

        lemma_transforms = collections.Counter()

        # Load the CoNLL-U file
        with open(path, "r", encoding="utf-8") if text is None else io.StringIO(text) as file:
            in_sentence = False
            for line in file:
                line = line.rstrip("\r\n")

                if line:
                    if self.RE_EXTRAS.match(line):
                        if in_sentence:
                            while len(self._extras) < len(self.factors[0].strings): self._extras.append([])
                            while len(self._extras[-1]) <= len(self.factors[0].strings[-1]):
                                self._extras[-1].append("")
                        else:
                            while len(self._extras) <= len(self.factors[0].strings): self._extras.append([])
                            if not len(self._extras[-1]): self._extras[-1].append("")
                        self._extras[-1][-1] += ("\n" if self._extras[-1][-1] else "") + line
                        continue

                    columns = line.split("\t")[1:]
                    for f in range(self.FACTORS):
                        factor = self.factors[f]
                        if not in_sentence:
                            factor.word_ids.append([])
                            factor.strings.append([])

                        word = columns[f]
                        factor.strings[-1].append(word)

                        # Add word to word_ids
                        if f == self.FORMS:
                            # For formw, we do not remap strings into IDs because the tokenizer will create the subwords IDs for us.
                            factor.word_ids[-1].append(0)
                        elif f == self.HEAD:
                            factor.word_ids[-1].append(int(word) if word != "_" else -1)
                        elif f == self.LEMMAS:
                            factor.word_ids[-1].append(0)
                            lemma_transforms[(columns[self.FORMS], word)] += 1
                        else:
                            if f == self.DEPREL and args.deprel == "universal":
                                word = word.split(":")[0]
                            if word not in factor.words_map:
                                if train_dataset is not None:
                                    word = "<unk>"
                                else:
                                    factor.words_map[word] = len(factor.words)
                                    factor.words.append(word)
                            factor.word_ids[-1].append(factor.words_map[word])
                    in_sentence = True
                else:
                    in_sentence = False
                    for factor in self.factors:
                        if len(factor.word_ids): factor.word_ids[-1] = np.array(factor.word_ids[-1], np.int32)

            # Also load the file for evaluation if it is not a training dataset
            if train_dataset is not None:
                file.seek(0, io.SEEK_SET)
                self.conllu_for_eval = latinpipe_evalatin24_eval.load_conllu(file)

        # Construct lemma rules
        self.finalize_lemma_rules(lemma_transforms, create_rules=train_dataset is None)

        # The dataset consists of a single treebank
        self.treebank_ranges = [(0, len(self))]
        self.treebank_ids = [treebank_id]

        # Create an empty tokenize cache
        self._tokenizer_cache = {}

    def __len__(self):
        return len(self.factors[0].strings)

    def save_mappings(self, path: str) -> None:
        mappings = UDDataset.__new__(UDDataset)
        mappings.factors = []
        for factor in self.factors:
            mappings.factors.append(UDDataset.Factor.__new__(UDDataset.Factor))
            mappings.factors[-1].words = factor.words
        with open(path, "wb") as mappings_file:
            pickle.dump(mappings, mappings_file, protocol=4)

    @staticmethod
    def from_mappings(path: str) -> Self:
        with open(path, "rb") as mappings_file:
            mappings = pickle.load(mappings_file)
        for factor in mappings.factors:
            factor.words_map = {word: i for i, word in enumerate(factor.words)}
        return mappings

    @staticmethod
    def create_lemma_rule(form: str, lemma: str) -> str:
        diff = difflib.SequenceMatcher(None, form.lower(), lemma.lower(), False)
        rule, in_prefix = [], True
        for tag, i1, i2, j1, j2 in diff.get_opcodes():
            if i2 > len(form) // 3 and in_prefix:
                in_prefix = False
                if tag == "equal":
                    mode, jd = "L" if lemma[j2 - 1].islower() else "U", j2 - 1
                    while jd > j1 and lemma[jd - 1].islower() == lemma[j2 - 1].islower(): jd -= 1
                    rule.extend(["l" if lemma[j].islower() else "u" for j in range(j1, jd)])
                    rule.extend(mode * (len(form) - i2 + 1))
                if tag in ["replace", "delete"]:
                    rule.extend("D" * (len(form) - i2 + 1))
                if tag in ["replace", "insert"]:
                    rule.extend("i" + lemma[j] for j in range(j1, j2))
            else:
                if tag == "equal":
                    rule.extend(["l" if lemma[j].islower() else "u" for j in range(j1, j2)])
                if tag in ["replace", "delete"]:
                    rule.extend("d" * (i2 - i1))
                if tag in ["replace", "insert"]:
                    rule.extend("i" + lemma[j] for j in range(j1, j2))
        return "".join(rule)

    @staticmethod
    def apply_lemma_rule(rule: str, form: str) -> str:
        def error():
            # print("Error: cannot decode lemma rule '{}' with form '{}', copying input.".format(rule, form))
            return form

        if rule == "<unk>":
            return form

        lemma, r, i = [], 0, 0
        while r < len(rule):
            if rule[r] == "i":
                if r + 1 == len(rule):
                    return error()
                r += 1
                lemma.append(rule[r])
            elif rule[r] == "d":
                i += 1
            elif rule[r] in ("l", "u"):
                if i == len(form):
                    return error()
                lemma.append(form[i].lower() if rule[r] == "l" else form[i].upper())
                i += 1
            elif rule[r] in ("L", "U", "D"):
                i2 = len(form)
                while r + 1 < len(rule) and rule[r + 1] == rule[r]:
                    r += 1
                    i2 -= 1
                if i2 < i:
                    return error()
                if rule[r] == "L":
                    lemma.extend(form[i:i2].lower())
                if rule[r] == "U":
                    lemma.extend(form[i:i2].upper())
                i = i2
            else:
                return error()
            r += 1
        if i != len(form) or not lemma:
            return error()
        return "".join(lemma)

    def finalize_lemma_rules(self, lemma_transforms: collections.Counter, create_rules: bool) -> None:
        forms, lemmas = self.factors[self.FORMS], self.factors[self.LEMMAS]

        # Generate all rules
        rules_merged, rules_all = collections.Counter(), {}
        for form, lemma in lemma_transforms:
            rule = self.create_lemma_rule(form, lemma)
            rules_all[(form, lemma)] = rule
            if create_rules:
                rules_merged[rule] += 1

        # Keep the rules that are used more than once
        if create_rules:
            for rule, count in rules_merged.items():
                if count > 1:
                    lemmas.words_map[rule] = len(lemmas.words)
                    lemmas.words.append(rule)

        # Store the rules in the dataset
        for i in range(len(forms.strings)):
            for j in range(len(forms.strings[i])):
                rule = rules_all.get((forms.strings[i][j], lemmas.strings[i][j]))
                lemmas.word_ids[i][j] = lemmas.words_map.get(rule, 0)

    def tokenize(self, tokenizer: transformers.PreTrainedTokenizer) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if tokenizer not in self._tokenizer_cache:
            assert tokenizer.cls_token_id is not None, "The tokenizer must have a CLS token"

            tokenized = tokenizer(self.factors[0].strings, add_special_tokens=True, is_split_into_words=True)

            tokens, word_indices = [], []
            for i, sentence in enumerate(tokenized.input_ids):
                offset = 0
                if not len(sentence) or sentence[0] != tokenizer.cls_token_id:
                    # Handle tokenizers that do not add CLS tokens, which we need for prediction
                    # of the root nodes during parsing. For such tokenizers, we added the CLS token
                    # manually already, but the build_inputs_with_special_tokens() might not have added it.
                    sentence = [tokenizer.cls_token_id] + sentence
                    offset = 1

                treebank_id = None
                for id_, (start, end) in zip(self.treebank_ids, self.treebank_ranges):
                    if start <= i < end:
                        treebank_id = id_
                if treebank_id is not None:
                    sentence.insert(1, tokenizer.additional_special_tokens_ids[treebank_id])
                    offset += 1

                tokens.append(np.array(sentence, dtype=np.int32))
                word_indices.append([(0, 0)])
                for j in range(len(self.factors[0].strings[i])):
                    span = tokenized.word_to_tokens(i, j)
                    word_indices[-1].append((offset + span.start, offset + span.end - 1))
                word_indices[-1] = np.array(word_indices[-1], dtype=np.int32)

            self._tokenizer_cache[tokenizer] = (tokens, word_indices)

        return self._tokenizer_cache[tokenizer]

    def write_sentence(self, output: io.TextIOBase, index: int, overrides: list = None) -> None:
        assert index < len(self.factors[0].strings), "Sentence index out of range"

        for i in range(len(self.factors[0].strings[index]) + 1):
            # Start by writing extras
            if index < len(self._extras) and i < len(self._extras[index]) and self._extras[index][i]:
                print(self._extras[index][i], file=output)
            if i == len(self.factors[0].strings[index]): break

            fields = []
            fields.append(str(i + 1))
            for f in range(self.FACTORS):
                factor = self.factors[f]
                field = factor.strings[index][i]

                # Overrides
                if overrides is not None and f < len(overrides) and overrides[f] is not None:
                    override = overrides[f][i]
                    if f == self.HEAD:
                        field = str(override) if override >= 0 else "_"
                    else:
                        field = factor.words[override]
                        if f == self.LEMMAS:
                            field = self.apply_lemma_rule(field, self.factors[self.FORMS].strings[index][i])
                fields.append(field)

            print("\t".join(fields), file=output)
        print(file=output)


class UDDatasetMerged(UDDataset):
    def __init__(self, datasets: list[UDDataset]):
        # Create factors and other variables
        self.factors = []
        for f in range(self.FACTORS):
            self.factors.append(self.Factor(None))

        lemma_transforms = collections.Counter()

        self.treebank_ranges, self.treebank_ids = [], []
        for dataset in datasets:
            assert len(dataset.treebank_ranges) == len(dataset.treebank_ids) == 1
            self.treebank_ranges.append((len(self), len(self) + len(dataset)))
            self.treebank_ids.append(dataset.treebank_ids[0])
            for s in range(len(dataset)):
                for f in range(self.FACTORS):
                    factor = self.factors[f]
                    factor.strings.append(dataset.factors[f].strings[s])
                    factor.word_ids.append([])
                    for i, word in enumerate(dataset.factors[f].strings[s]):
                        if f == self.FORMS:
                            # We do not remap strings into IDs because the tokenizer will create the subwords IDs for us.
                            factor.word_ids[-1].append(0)
                        if f == self.HEAD:
                            factor.word_ids[-1].append(word)
                        elif f == self.LEMMAS:
                            factor.word_ids[-1].append(0)
                            lemma_transforms[(dataset.factors[self.FORMS].strings[s][i], word)] += 1
                        else:
                            if word not in factor.words_map:
                                factor.words_map[word] = len(factor.words)
                                factor.words.append(word)
                            factor.word_ids[-1].append(factor.words_map[word])
                    self.factors[f].word_ids[-1] = np.array(self.factors[f].word_ids[-1], np.int32)

        # Construct lemma rules
        self.finalize_lemma_rules(lemma_transforms, create_rules=True)

        # Create an empty tokenize cache
        self._tokenizer_cache = {}


class TorchUDDataset(torch.utils.data.Dataset):
    def __init__(self, ud_dataset: UDDataset, tokenizers: list[transformers.PreTrainedTokenizer], args: argparse.Namespace, training: bool):
        self.ud_dataset = ud_dataset
        self.training = training
        self._outputs_to_input = [args.tags.index(tag) for tag in args.embed_tags]

        self._inputs = [ud_dataset.tokenize(tokenizer) for tokenizer in tokenizers]
        self._outputs = [ud_dataset.factors[tag].word_ids for tag in args.tags]
        if args.parse:
            self._outputs.append(ud_dataset.factors[ud_dataset.HEAD].word_ids)
            self._outputs.append(ud_dataset.factors[ud_dataset.DEPREL].word_ids)

        # Trim the sentences if needed
        if training and args.max_train_sentence_len:
            trimmed_sentences = 0
            for index in range(len(self)):  # Over sentences
                max_words, need_trimming = None, False
                for tokens, word_indices in self._inputs:  # Over transformers
                    if max_words is None:
                        max_words = len(word_indices[index])
                    while word_indices[index][max_words - 1, 1] >= args.max_train_sentence_len:
                        max_words -= 1
                        need_trimming = True
                assert max_words >= 2, "Sentence too short after trimming"

                if need_trimming:
                    for tokens, word_indices in self._inputs:  # Over transformers
                        tokens[index] = tokens[index][:word_indices[index][max_words - 1, 1] + 1]
                        word_indices[index] = word_indices[index][:max_words]

                    for output in self._outputs:
                        output[index] = output[index][:max_words - 1]  # No CLS tokens in outputs
                    if args.parse:
                        self._outputs[-2][index] = np.array([head if head < max_words else -1 for head in self._outputs[-2][index]], np.int32)

                    trimmed_sentences += 1
            if trimmed_sentences:
                print("Trimmed {} out of {} sentences".format(trimmed_sentences, len(self)))

    def __len__(self):
        return len(self.ud_dataset)

    def __getitem__(self, index: int):
        inputs = []
        for tokens, word_indices in self._inputs:
            inputs.append(torch.from_numpy(tokens[index]))
            inputs.append(torch.from_numpy(word_indices[index]))
        for i in self._outputs_to_input:
            inputs.append(torch.from_numpy(self._outputs[i][index]))

        outputs = []
        for output in self._outputs:
            outputs.append(torch.from_numpy(output[index]))

        return inputs, outputs


class TorchUDDataLoader(torch.utils.data.DataLoader):
    class MergedDatasetSampler(torch.utils.data.Sampler):
        def __init__(self, ud_dataset: UDDataset, args: argparse.Namespace):
            self._treebank_ranges = ud_dataset.treebank_ranges
            self._sentences_per_epoch = args.steps_per_epoch * args.batch_size
            self._generator = torch.Generator().manual_seed(args.seed)

            treebank_weights = np.array([r[1] - r[0] for r in self._treebank_ranges], np.float32)
            treebank_weights = treebank_weights ** args.train_sampling_exponent
            treebank_weights /= np.sum(treebank_weights)
            self._treebank_sizes = np.array(treebank_weights * self._sentences_per_epoch, np.int32)
            self._treebank_sizes[:self._sentences_per_epoch - np.sum(self._treebank_sizes)] += 1
            self._treebank_indices = [[] for _ in self._treebank_ranges]

        def __len__(self):
            return self._sentences_per_epoch

        def __iter__(self):
            indices = []
            for i in range(len(self._treebank_ranges)):
                required = self._treebank_sizes[i]
                while required:
                    if not len(self._treebank_indices[i]):
                        self._treebank_indices[i] = self._treebank_ranges[i][0] + torch.randperm(
                            self._treebank_ranges[i][1] - self._treebank_ranges[i][0], generator=self._generator)
                    indices.append(self._treebank_indices[i][:required])
                    required -= min(len(self._treebank_indices[i]), required)
            indices = torch.concatenate(indices, axis=0)
            return iter(indices[torch.randperm(len(indices), generator=self._generator)])

    def _collate_fn(self, batch):
        inputs, outputs = zip(*batch)

        batch_inputs = []
        for sequences in zip(*inputs):
            batch_inputs.append(torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=-1))

        batch_outputs = []
        for output in zip(*outputs):
            batch_outputs.append(torch.nn.utils.rnn.pad_sequence(output, batch_first=True, padding_value=-1))

        batch_weights = [batch_output != -1 for batch_output in batch_outputs]

        return tuple(batch_inputs), tuple(batch_outputs), tuple(batch_weights)

    def __init__(self, dataset: TorchUDDataset, args: argparse.Namespace, **kwargs):
        sampler = None
        if dataset.training:
            if len(dataset.ud_dataset.treebank_ranges) == 1:
                sampler = torch.utils.data.RandomSampler(dataset, generator=torch.Generator().manual_seed(args.seed))
            else:
                assert args.steps_per_epoch is not None, "Steps per epoch must be specified when training on multiple treebanks"
                sampler = self.MergedDatasetSampler(dataset.ud_dataset, args)
        super().__init__(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=self._collate_fn, **kwargs)


class LatinPipeModel(keras.Model):
    class HFTransformerLayer(keras.layers.Layer):
        def __init__(self, transformer: transformers.PreTrainedModel, subword_combination: str, word_masking: float = None, mask_token_id: int = None, **kwargs):
            super().__init__(**kwargs)
            self._transformer = transformer
            self._subword_combination = subword_combination
            self._word_masking = word_masking
            self._mask_token_id = mask_token_id

        def call(self, inputs, word_indices, training=None):
            if training and self._word_masking:
                mask = keras.ops.cast(keras.random.uniform(keras.ops.shape(inputs), dtype="float32") < self._word_masking, inputs.dtype)
                inputs = (1 - mask) * inputs + mask * self._mask_token_id
            if (training or False) != self._transformer.training:
                self._transformer.train(training or False)
            if self._subword_combination != "last":
                first_subwords = keras.ops.take_along_axis(
                    self._transformer(keras.ops.maximum(inputs, 0), attention_mask=inputs > -1).last_hidden_state,
                    keras.ops.expand_dims(keras.ops.maximum(word_indices[..., 0], 0), axis=-1),
                    axis=1,
                )
            if self._subword_combination != "first":
                last_subwords = keras.ops.take_along_axis(
                    self._transformer(keras.ops.maximum(inputs, 0), attention_mask=inputs > -1).last_hidden_state,
                    keras.ops.expand_dims(keras.ops.maximum(word_indices[..., 1], 0), axis=-1),
                    axis=1,
                )
            if self._subword_combination == "first":
                return first_subwords
            elif self._subword_combination == "last":
                return last_subwords
            elif self._subword_combination == "sum":
                return first_subwords + last_subwords
            elif self._subword_combination == "concat":
                return keras.ops.concatenate([first_subwords, last_subwords], axis=-1)
            else:
                raise ValueError("Unknown subword combination '{}'".format(self._subword_combination))

    class LSTMTorch(keras.layers.Layer):
        def __init__(self, units: int, **kwargs):
            super().__init__(**kwargs)
            self._units = units

        def build(self, input_shape):
            self._lstm = torch.nn.LSTM(input_shape[-1], self._units, batch_first=True, bidirectional=True)

        def call(self, inputs, lengths):
            packed_result, _ = self._lstm.module(torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False))
            unpacked_result = torch.nn.utils.rnn.unpack_sequence(packed_result)
            return torch.nn.utils.rnn.pad_sequence(unpacked_result, batch_first=True, padding_value=0)

    class GRUTorch(keras.layers.Layer):
        def __init__(self, units: int, **kwargs):
            super().__init__(**kwargs)
            self._units = units

        def build(self, input_shape):
            self._gru = torch.nn.GRU(input_shape[-1], self._units, batch_first=True, bidirectional=True)

        def call(self, inputs, lengths):
            packed_result, _ = self._gru(torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False))
            unpacked_result = torch.nn.utils.rnn.unpack_sequence(packed_result)
            return torch.nn.utils.rnn.pad_sequence(unpacked_result, batch_first=True, padding_value=0)

    class ParsingHead(keras.layers.Layer):
        def __init__(self, num_deprels: int, task_hidden_layer: int, parse_attention_dim: int, dropout: float, **kwargs):
            super().__init__(**kwargs)
            self._head_queries_hidden = keras.layers.Dense(task_hidden_layer, activation="relu")
            self._head_queries_output = keras.layers.Dense(parse_attention_dim)
            self._head_keys_hidden = keras.layers.Dense(task_hidden_layer, activation="relu")
            self._head_keys_output = keras.layers.Dense(parse_attention_dim)
            self._deprel_hidden = keras.layers.Dense(task_hidden_layer, activation="relu")
            self._deprel_output = keras.layers.Dense(num_deprels)
            self._dropout = keras.layers.Dropout(dropout)

        def call(self, embeddings, embeddings_wo_root, embeddings_mask):
            head_queries = self._head_queries_output(self._dropout(self._head_queries_hidden(embeddings_wo_root)))
            head_keys = self._head_keys_output(self._dropout(self._head_keys_hidden(embeddings)))
            head_scores = keras.ops.matmul(head_queries, keras.ops.transpose(head_keys, axes=[0, 2, 1])) / keras.ops.sqrt(head_queries.shape[-1])

            head_scores_mask = keras.ops.cast(keras.ops.expand_dims(embeddings_mask, axis=1), head_scores.dtype)
            head_scores = head_scores * head_scores_mask - 1e9 * (1 - head_scores_mask)

            predicted_heads = keras.ops.argmax(head_scores, axis=-1)
            predicted_head_embeddings = keras.ops.take_along_axis(embeddings, keras.ops.expand_dims(predicted_heads, axis=-1), axis=1)
            deprel_hidden = keras.ops.concatenate([embeddings_wo_root, predicted_head_embeddings], axis=-1)
            deprel_scores = self._deprel_output(self._dropout(self._deprel_hidden(deprel_hidden)))

            return head_scores, deprel_scores

    class SparseCategoricalCrossentropyWithLabelSmoothing(keras.losses.Loss):
        def __init__(self, from_logits: bool, label_smoothing: float, **kwargs):
            super().__init__(**kwargs)
            self._from_logits = from_logits
            self._label_smoothing = label_smoothing

        def call(self, y_true, y_pred):
            y_gold = keras.ops.one_hot(keras.ops.maximum(y_true, 0), y_pred.shape[-1])
            if self._label_smoothing:
                y_pred_mask = keras.ops.cast(y_pred > -1e9, y_pred.dtype)
                y_gold = y_gold * (1 - self._label_smoothing) + y_pred_mask / keras.ops.sum(y_pred_mask, axis=-1, keepdims=True) * self._label_smoothing
            return keras.losses.categorical_crossentropy(y_gold, y_pred, from_logits=self._from_logits)

    def __init__(self, dataset: UDDataset, args: argparse.Namespace):
        self._dataset = dataset
        self._args = args

        # Create the transformer models
        self._tokenizers, self._transformers = [], []
        for name in args.transformers:
            self._tokenizers.append(transformers.AutoTokenizer.from_pretrained(name, add_prefix_space=True))

            transformer, transformer_opts = transformers.AutoModel, {}
            if "mt5" in name.lower():
                transformer = transformers.MT5EncoderModel
            if name.endswith(("LaTa", "PhilTa")):
                transformer = transformers.T5EncoderModel
            if name.endswith(("LaBerta", "PhilBerta")):
                transformer_opts["add_pooling_layer"] = False

            if args.load:
                transformer = transformer.from_config(transformers.AutoConfig.from_pretrained(name), **transformer_opts)
            else:
                transformer = transformer.from_pretrained(name, **transformer_opts)

            # Create additional tokens
            additional_tokens = {}
            if args.treebank_ids:
                additional_tokens["additional_special_tokens"] = ["[TREEBANK_ID_{}]".format(i) for i in range(len(dataset.treebank_ids))]
            if self._tokenizers[-1].cls_token_id is None:  # Generate CLS token if not present (for representing sentence root in parsing).
                additional_tokens["cls_token"] = "[CLS]"
            if additional_tokens:
                self._tokenizers[-1].add_special_tokens(additional_tokens)
                transformer.resize_token_embeddings(len(self._tokenizers[-1]))
            if args.treebank_ids:
                assert len(self._tokenizers[-1].additional_special_tokens) == len(dataset.treebank_ids)

            self._transformers.append(self.HFTransformerLayer(transformer, args.subword_combination, args.word_masking, self._tokenizers[-1].mask_token_id))

        # Create the network
        inputs = []
        for _ in args.transformers:
            inputs.extend([keras.layers.Input(shape=[None], dtype="int32"), keras.layers.Input(shape=[None, 2], dtype="int32")])
        for _ in args.embed_tags:
            inputs.append(keras.layers.Input(shape=[None], dtype="int32"))

        # Run the transformer models
        embeddings = []
        for tokens, word_indices, transformer in zip(inputs[::2], inputs[1::2], self._transformers):
            embeddings.append(transformer(tokens, word_indices))
        embeddings = keras.layers.Concatenate(axis=-1)(embeddings)
        embeddings = keras.layers.Dropout(args.dropout)(embeddings)

        # Heads for the tagging tasks
        outputs = []
        for tag in args.tags:
            hidden = keras.layers.Dense(args.task_hidden_layer, activation="relu")(embeddings[:, 1:])
            hidden = keras.layers.Dropout(args.dropout)(hidden)
            outputs.append(keras.layers.Dense(len(dataset.factors[tag].words))(hidden))

        # Head for parsing
        if args.parse:
            if args.embed_tags:
                all_embeddings = [embeddings]
                for factor, input_tags in zip(args.embed_tags, inputs[-len(args.embed_tags):]):
                    embedding_layer = keras.layers.Embedding(len(dataset.factors[factor].words) + 1, 256)
                    all_embeddings.append(keras.layers.Dropout(args.dropout)(embedding_layer(keras.ops.pad(input_tags + 1, [(0, 0), (1, 0)]))))
                embeddings = keras.ops.concatenate(all_embeddings, axis=-1)

            for i in range(args.rnn_layers):
                if args.rnn_type in ["LSTM", "GRU"]:
                    hidden = keras.layers.Bidirectional(getattr(keras.layers, args.rnn_type)(args.rnn_dim, return_sequences=True))(embeddings, mask=inputs[1][..., 0] > -1)
                elif args.rnn_type in ["LSTMTorch", "GRUTorch"]:
                    hidden = getattr(self, args.rnn_type)(args.rnn_dim)(embeddings, keras.ops.sum(inputs[1][..., 0] > -1, axis=-1))
                hidden = keras.layers.Dropout(args.dropout)(hidden)
                embeddings = hidden + (embeddings if i else 0)

            outputs.extend(self.ParsingHead(
                len(dataset.factors[dataset.DEPREL].words), args.task_hidden_layer, args.parse_attention_dim, args.dropout,
            )(embeddings, embeddings[:, 1:], inputs[1][..., 0] > -1))

        super().__init__(inputs=inputs, outputs=outputs)
        if args.load:
            self.load_weights(args.load[0])

    def compile(self, epoch_batches: int, frozen: bool):
        args = self._args

        for transformer in self._transformers:
            transformer.trainable = not frozen

        if frozen:
            schedule = 1e-3
        else:
            schedule = keras.optimizers.schedules.CosineDecay(
                0. if args.learning_rate_warmup else args.learning_rate,
                args.epochs * epoch_batches - args.learning_rate_warmup,
                alpha=0.0 if args.learning_rate_decay != "none" else 1.0,
                warmup_target=args.learning_rate if args.learning_rate_warmup else None,
                warmup_steps=args.learning_rate_warmup,
            )
        if args.optimizer == "adam":
            optimizer = keras.optimizers.Adam(schedule)
        elif args.optimizer == "adafactor":
            optimizer = keras.optimizers.Adafactor(schedule)
        else:
            raise ValueError("Unknown optimizer '{}'".format(args.optimizer))
        super().compile(
            optimizer=optimizer,
            loss=self.SparseCategoricalCrossentropyWithLabelSmoothing(from_logits=True, label_smoothing=args.label_smoothing),
        )

    @property
    def tokenizers(self) -> list[transformers.PreTrainedTokenizer]:
        return self._tokenizers

    def predict(self, dataloader: TorchUDDataLoader, save_as: str|None = None, args_override: argparse.Namespace|None = None) -> str:
        ud_dataset = dataloader.dataset.ud_dataset
        args = self._args if args_override is None else args_override
        conllu, sentence = io.StringIO(), 0

        for batch_inputs, _, _ in dataloader:
            predictions = self.predict_on_batch(batch_inputs)
            for b in range(len(batch_inputs[0])):
                sentence_len = len(ud_dataset.factors[ud_dataset.FORMS].strings[sentence])
                overrides = [None] * ud_dataset.FACTORS
                for tag, prediction in zip(args.tags, predictions):
                    overrides[tag] = np.argmax(prediction[b, :sentence_len], axis=-1)
                if args.parse:
                    heads, deprels = predictions[-2:]
                    padded_heads = np.zeros([sentence_len + 1, sentence_len + 1], dtype=np.float64)
                    padded_heads[1:] = heads[b, :sentence_len, :sentence_len + 1]
                    padded_heads[1:] -= np.max(padded_heads[1:], axis=-1, keepdims=True)
                    padded_heads[1:] -= np.log(np.sum(np.exp(padded_heads[1:]), axis=-1, keepdims=True))
                    if args.single_root:
                        selected_root = 1 + np.argmax(padded_heads[1:, 0])
                        padded_heads[:, 0] = np.nan
                        padded_heads[selected_root, 0] = 0
                    chosen_heads, _ = ufal.chu_liu_edmonds.chu_liu_edmonds(padded_heads)
                    overrides[ud_dataset.HEAD] = chosen_heads[1:]
                    overrides[ud_dataset.DEPREL] = np.argmax(deprels[b, :sentence_len], axis=-1)
                ud_dataset.write_sentence(conllu, sentence, overrides)
                sentence += 1

        conllu = conllu.getvalue()
        if save_as is not None:
            os.makedirs(os.path.dirname(save_as), exist_ok=True)
            with open(save_as, "w", encoding="utf-8") as conllu_file:
                conllu_file.write(conllu)
        return conllu

    def evaluate(self, dataloader: TorchUDDataLoader, save_as: str|None = None, args_override: argparse.Namespace|None = None) -> tuple[str, dict[str, float]]:
        conllu = self.predict(dataloader, save_as=save_as, args_override=args_override)
        evaluation = latinpipe_evalatin24_eval.evaluate(dataloader.dataset.ud_dataset.conllu_for_eval, latinpipe_evalatin24_eval.load_conllu(io.StringIO(conllu)))
        if save_as is not None:
            os.makedirs(os.path.dirname(save_as), exist_ok=True)
            with open(save_as + ".eval", "w", encoding="utf-8") as eval_file:
                for metric, score in evaluation.items():
                    print("{}: {:.2f}%".format(metric, 100 * score.f1), file=eval_file)
        return conllu, evaluation


class LatinPipeModelEnsemble:
    def __init__(self, latinpipe_model: LatinPipeModel, args: argparse.Namespace):
        self._latinpipe_model = latinpipe_model
        self._args = args

    def predict(self, dataloader: TorchUDDataLoader, save_as: str|None = None) -> str:
        def log_softmax(logits):
            logits -= np.max(logits, axis=-1, keepdims=True)
            logits -= np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
            return logits
        ud_dataset = dataloader.dataset.ud_dataset

        # First compute all predictions
        overrides = [[0] * len(ud_dataset) if tag in self._args.tags + ([ud_dataset.HEAD, ud_dataset.DEPREL] if self._args.parse else []) else None
                     for tag in range(ud_dataset.FACTORS)]
        for path in self._args.load:
            self._latinpipe_model.load_weights(path)
            sentence = 0
            for batch_inputs, _, _ in dataloader:
                predictions = self._latinpipe_model.predict_on_batch(batch_inputs)
                for b in range(len(batch_inputs[0])):
                    sentence_len = len(ud_dataset.factors[ud_dataset.FORMS].strings[sentence])
                    for tag, prediction in zip(self._args.tags, predictions):
                        overrides[tag][sentence] += log_softmax(prediction[b, :sentence_len])
                    if self._args.parse:
                        overrides[ud_dataset.HEAD][sentence] += log_softmax(predictions[-2][b, :sentence_len, :sentence_len + 1])
                        overrides[ud_dataset.DEPREL][sentence] += log_softmax(predictions[-1][b, :sentence_len])
                    sentence += 1

        # Predict the most likely class and generate CoNLL-U output
        conllu = io.StringIO()
        for sentence in range(len(ud_dataset)):
            sentence_overrides = [None] * ud_dataset.FACTORS
            for tag in self._args.tags:
                sentence_overrides[tag] = np.argmax(overrides[tag][sentence], axis=-1)
            if self._args.parse:
                padded_heads = np.pad(overrides[ud_dataset.HEAD][sentence], [(1, 0), (0, 0)]).astype(np.float64)
                if self._args.single_root:
                    selected_root = 1 + np.argmax(padded_heads[1:, 0])
                    padded_heads[:, 0] = np.nan
                    padded_heads[selected_root, 0] = 0
                chosen_heads, _ = ufal.chu_liu_edmonds.chu_liu_edmonds(padded_heads)
                sentence_overrides[ud_dataset.HEAD] = chosen_heads[1:]
                sentence_overrides[ud_dataset.DEPREL] = np.argmax(overrides[ud_dataset.DEPREL][sentence], axis=-1)
            ud_dataset.write_sentence(conllu, sentence, sentence_overrides)

        conllu = conllu.getvalue()
        if save_as is not None:
            os.makedirs(os.path.dirname(save_as), exist_ok=True)
            with open(save_as, "w", encoding="utf-8") as conllu_file:
                conllu_file.write(conllu)
        return conllu

    def evaluate(self, dataloader: TorchUDDataLoader, save_as: str|None = None) -> tuple[str, dict[str, float]]:
        return LatinPipeModel.evaluate(self, dataloader, save_as=save_as)


def main(params: list[str] | None = None) -> None:
    args = parser.parse_args(params)

    # If supplied, load configuration from a trained model
    if args.load:
        with open(os.path.join(os.path.dirname(args.load[0]), "options.json"), mode="r") as options_file:
            args = argparse.Namespace(**{k: v for k, v in json.load(options_file).items() if k not in [
                "dev", "exp", "load", "test", "threads", "verbose"]})
            args = parser.parse_args(params, namespace=args)
    else:
        assert args.train, "Either --load or --train must be set."
        assert args.transformers, "At least one transformer must be specified."

        # Post-process arguments
        args.embed_tags = [UDDataset.FACTORS_MAP[tag] for tag in args.embed_tags.split(",") if tag]
        args.tags = [UDDataset.FACTORS_MAP[tag] for tag in args.tags.split(",") if tag]
        args.script = os.path.basename(__file__)

        # Create logdir
        args.logdir = os.path.join("logs", "{}{}-{}-{}-s{}".format(
            args.exp + "-" if args.exp else "",
            os.path.splitext(os.path.basename(globals().get("__file__", "notebook")))[0],
            os.environ.get("SLURM_JOB_ID", ""),
            datetime.datetime.now().strftime("%y%m%d_%H%M%S"),
            args.seed,
            # ",".join(("{}={}".format(
            #     re.sub("(.)[^_]*_?", r"\1", k),
            #     ",".join(re.sub(r"^.*/", "", str(x)) for x in ((v if len(v) <= 1 else [v[0], "..."]) if isinstance(v, list) else [v])),
            # ) for k, v in sorted(vars(args).items()) if k not in ["dev", "exp", "load", "test", "threads", "verbose"]))
        ))
        print(json.dumps(vars(args), sort_keys=True, ensure_ascii=False, indent=2))
        os.makedirs(args.logdir, exist_ok=True)
        with open(os.path.join(args.logdir, "options.json"), mode="w") as options_file:
            json.dump(vars(args), options_file, sort_keys=True, ensure_ascii=False, indent=2)

    # Set the random seed and the number of threads
    keras.utils.set_random_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Load the data
    if args.treebank_ids and max(len(args.train), len(args.dev), len(args.test)) > 1:
        print("WARNING: With treebank_ids, treebanks must always be in the same position in the train/dev/test.")
    if args.load:
        train = UDDataset.from_mappings(os.path.join(os.path.dirname(args.load[0]), "mappings.pkl"))
    else:
        train = UDDatasetMerged([UDDataset(path, args, treebank_id=i if args.treebank_ids else None) for i, path in enumerate(args.train)])
        train.save_mappings(os.path.join(args.logdir, "mappings.pkl"))
    devs = [UDDataset(path, args, treebank_id=i if args.treebank_ids else None, train_dataset=train) for i, path in enumerate(args.dev)]
    tests = [UDDataset(path, args, treebank_id=i if args.treebank_ids else None, train_dataset=train) for i, path in enumerate(args.test)]

    # Create the model
    model = LatinPipeModel(train, args)

    # Create the dataloaders
    if not args.load:
        train_dataloader = TorchUDDataLoader(TorchUDDataset(train, model.tokenizers, args, training=True), args)
    dev_dataloaders = [TorchUDDataLoader(TorchUDDataset(dataset, model.tokenizers, args, training=False), args) for dataset in devs]
    test_dataloaders = [TorchUDDataLoader(TorchUDDataset(dataset, model.tokenizers, args, training=False), args) for dataset in tests]

    # Perform prediction if requested
    if args.load:
        if len(args.load) > 1:
            model = LatinPipeModelEnsemble(model, args)
        for dataloader in dev_dataloaders:
            model.evaluate(dataloader, save_as=os.path.splitext(
                os.path.join(args.exp, os.path.basename(dataloader.dataset.ud_dataset.path)) if args.exp else dataloader.dataset.ud_dataset.path
            )[0] + ".predicted.conllu")
        for dataloader in test_dataloaders:
            model.predict(dataloader, save_as=os.path.splitext(
                os.path.join(args.exp, os.path.basename(dataloader.dataset.ud_dataset.path)) if args.exp else dataloader.dataset.ud_dataset.path
            )[0] + ".predicted.conllu")
        return

    # Train the model
    class Evaluator(keras.callbacks.Callback):
        def __init__(self, wandb_log):
            super().__init__()
            self._wandb_log = wandb_log
            self._metrics = [["", "Lemmas", "UPOS", "XPOS", "UFeats"][tag] for tag in args.tags] + (["UAS", "LAS"] if args.parse else [])

        def on_epoch_end(self, epoch, logs=None):
            logs["learning_rate"] = keras.ops.convert_to_numpy(model.optimizer.learning_rate)
            for dataloader in dev_dataloaders + (test_dataloaders if epoch + 1 == args.epochs + args.epochs_frozen else []):
                _, metrics = model.evaluate(dataloader, save_as=os.path.splitext(
                    os.path.join(args.logdir, os.path.basename(dataloader.dataset.ud_dataset.path))
                )[0] + ".{:02d}.conllu".format(epoch + 1))
                for metric, score in metrics.items():
                    if metric in self._metrics:
                        logs["{}_{}".format(os.path.splitext(os.path.basename(dataloader.dataset.ud_dataset.path))[0], metric)] = 100 * score.f1

            aggregations = {"la_ud213": [("la_ittb-ud", 390_787), ("la_llct-ud", 194_143), ("la_proiel-ud", 177_558),
                                         ("la_udante-ud", 30_450), ("la_perseus-ud", 16_486)]}
            for split in ["dev", "test"]:
                for metric in self._metrics:
                    for aggregation, parts in aggregations.items():
                        values = [logs.get("{}-{}_{}".format(part, split, metric), None) for part, _ in parts]
                        if all(value is not None for value in values):
                            logs["{}-{}_{}".format(aggregation, split, metric)] = np.mean(values)
                            logs["{}-sqrt-{}_{}".format(aggregation, split, metric)] = np.average(values, weights=[size**0.5 for _, size in parts])

            if self._wandb_log:
                self._wandb_log(logs, step=epoch + 1, commit=True)

    wandb_log = None
    if args.wandb:
        import wandb
        wandb.init(project="ufal-evalatin2024", name=args.exp, config=vars(args))
        wandb_log = wandb.log
    evaluator = Evaluator(wandb_log)
    if args.epochs_frozen:
        model.compile(len(train_dataloader), frozen=True)
        model.fit(train_dataloader, epochs=args.epochs_frozen, verbose=args.verbose, callbacks=[evaluator])
    if args.epochs:
        model.compile(len(train_dataloader), frozen=False)
        model.fit(train_dataloader, initial_epoch=args.epochs_frozen, epochs=args.epochs_frozen + args.epochs, verbose=args.verbose, callbacks=[evaluator])
    if args.save_checkpoint:
        model.save_weights(os.path.join(args.logdir, "model.weights.h5"))


if __name__ == "__main__":
    main([] if "__file__" not in globals() else None)
