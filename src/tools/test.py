#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import math
import MeCab

def tsize(ysm):
    ysm = ["".join(ys) for ys in ysm]
    tmp = {}
    for i, word in enumerate(ysm):
        if word not in tmp:
            tmp[word] = 1.0
        else:
            tmp[word] += 1.0
    #for k, v in sorted(tmp.items(), key=lambda x: -x[1]):
    #    print(str(k) + "," + str(v))
    return len(tmp)


def match(ysm, ysr):
    ysm = ["".join(ys) for ys in ysm]
    ysr = ["".join(ys) for ys in ysr]
    cnt = 0.0
    for yr, ym in zip(ysr, ysm):
        cnt += 1.0 if (yr == ym) else 0.0
    return cnt / len(ysr)


def make_ngram(ys, n):
    s = ""
    ngram = []
    for i in range(len(ys) - (n - 1)):
        for j in range(n):
            s += ys[i + j]
        ngram.append(s)
        s = ""
    return ngram


def intersect_list(lst1, lst2):
    arr = []
    lst = lst1.copy()
    for element in lst2:
        try:
            lst.remove(element)
        except ValueError:
            pass
        else:
            arr.append(element)
    return arr

def bleu(ysm, ysr, n):
    numer = [0.0] * n
    denom = [0.0] * n
    ym_len = 0.0
    yr_len = 0.0
    for (ym, yr) in zip(ysm, ysr):
        ym_len += len(ym)
        yr_len += len(yr)
        for i in range(1, n + 1):
            ngramM = make_ngram(ym, i)
            ngramR = make_ngram(yr, i)
            denom[i - 1] += len(ngramM)
            numer[i - 1] += len(intersect_list(ngramM, ngramR))
    exp_tmp = 0.0
    for i in range(n - 1, -1, -1):
        pn = numer[i] / denom[i] if denom[i] != 0.0 else 0.0
        # print(str(pn)+"="+str(numer[i])+"/"+str(denom[i]))
        if pn == 0.0:
            n -= 1
            continue
        exp_tmp += (1 / n) * math.log(pn)
    # print('1/'+str(n))
    BP = 1.0
    if ym_len < yr_len:
        BP = math.exp(1 - (yr_len / ym_len))
    BLEU = BP * math.exp(exp_tmp)
    return BLEU


def main(argv):

    if len(argv) < 3:
        print("python " + str(argv[0] + " ms" + " rs"))
        sys.exit(0)
    m = MeCab.Tagger("-Owakati")

    ysml = [
        ys.replace(" ", "<split>").split("<split>")
        for ys in [
            m.parse(ys).replace("\n", "").strip()
            for ys in open(argv[1]).read().replace(" ", "").strip().split("\n")
        ]
    ]
    ysrl = [
        ys.replace(" ", "<split>").split("<split>")
        for ys in [
            m.parse(ys).replace("\n", "").strip()
            for ys in open(argv[2]).read().replace(" ", "").strip().split("\n")
        ]
    ]
    ysm = open(argv[1]).read().replace(" ", "").strip().split("\n")
    ysr = open(argv[2]).read().replace(" ", "").strip().split("\n")
    print(match(ysm, ysr),',',bleu(ysml, ysrl, 4),',',tsize(ysm))
    #print("BLEU : " + str(bleu(ysml, ysrl, 4)))
    #print("一致率 : " + str(match(ysm, ysr)))
    #print("種類数 : " + str(tsize(ysm)))


if __name__ == "__main__":

    argv = sys.argv
    main(argv)
