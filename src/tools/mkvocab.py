#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import pickle

rvocab = {}
vocab = {}


def load_data(filename):
    global vocab
    words = open(filename).read().replace("\n", "<eos> ").strip().split()
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
            rvocab[vocab[word]] = word


def main(argv):

    if len(argv) < 2:
        print("python mkvocab.py file...")
        sys.exit(0)

    for i in range(1, len(argv)):
        load_data(argv[i])
    with open("vocab.dump", "wb") as output:
        pickle.dump(vocab, output)
    with open("rvocab.dump", "wb") as output:
        pickle.dump(rvocab, output)


if __name__ == "__main__":
    argv = sys.argv
    main(argv)
