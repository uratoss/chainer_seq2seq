#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import math
import MeCab

def main(argv):

    ys = open(argv[1]).read().replace(" ", "").strip().split("\n")
    for yy in ys:
        print(yy)

if __name__ == "__main__":

    argv = sys.argv
    main(argv)
