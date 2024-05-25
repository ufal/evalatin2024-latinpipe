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

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="CoNLL-U file to split")
    parser.add_argument("train", type=str, help="CoNLL-U file to write training data to")
    parser.add_argument("dev", type=str, help="CoNLL-U file to write development data to")
    parser.add_argument("--dev_size", type=float, default=0.1, help="Size of the development data")
    args = parser.parse_args()

    sentences = []
    with open(args.source, "r", encoding="utf-8") as source:
        sentence = []
        for line in source:
            sentence.append(line)
            if not line.rstrip("\r\n"):
                sentences.append("".join(sentence))
                sentence = []
        assert not sentence, "Missing empty line after the last sentence"

    dev_indices = set(np.random.RandomState(42).choice(len(sentences), int(len(sentences) * args.dev_size), replace=False))

    with open(args.train, "w", encoding="utf-8") as train:
        with open(args.dev, "w", encoding="utf-8") as dev:
            for i, sentence in enumerate(sentences):
                (dev if i in dev_indices else train).write(sentence)
