#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import math
import numpy as np
import chainer
from chainer import cuda, Function, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


def load_data(filename, vocab):
    words = open(filename).read().replace("\n", "<eos> ").strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        dataset[i] = vocab[word]
    return dataset


class seq2seq(chainer.Chain):
    def __init__(self, lay, vocab, k, dout):
        super(seq2seq, self).__init__(
            embedx=L.EmbedID(vocab, k),
            embedy=L.EmbedID(vocab, k),
            encoder=L.NStepLSTM(lay, k, k, dout),
            decoder=L.NStepLSTM(lay, k, k, dout),
            W=L.Linear(k, vocab),
        )

    def __call__(self, hx, cx, xs, ys, batch_size, vocab):
        accum_loss = None
        # エンコーダ側の学習
        embs_x = [self.embedx(x) for x in xs]
        hy, cy, _ = self.encoder(hx, cx, embs_x)

        # デコーダ側の学習
        # eosを付与
        # ysを入力と教師信号に分ける
        eos = np.array([vocab["<eos>"]], dtype=np.int32)
        ys_in = [F.concat((eos, y), axis=0) for y in ys]
        ys_out = [F.concat((y, eos), axis=0) for y in ys]

        embs_y_in = [self.embedy(y) for y in ys_in]
        _, _, embs_y_out = self.decoder(hy, cy, embs_y_in)

        # 損失を計算する
        for emb_y_out, y_out in zip(embs_y_out, ys_out):
            loss = F.softmax_cross_entropy(self.W(emb_y_out), y_out)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        accum_loss = accum_loss / batch_size
        return accum_loss


def generate(model, xs, layer, demb, vocab, rvocab):
    with chainer.using_config("enable_backprop", False), chainer.using_config(
        "train", False
    ):
        # 初期値を生成
        hx = chainer.Variable(np.zeros((layer, len(xs), demb), dtype=np.float32))
        cx = chainer.Variable(np.zeros((layer, len(xs), demb), dtype=np.float32))
        accum_loss = None
        # エンコーダ側の処理
        embs_x = [model.embedx(x) for x in xs]
        hy, cy, _ = model.encoder(hx, cx, embs_x)

        # デコーダ側の処理
        # eosを入力して始める
        wid = vocab["<eos>"]
        ys = []
        loop = 0
        while True:
            emb_y_in = [
                model.embedy(chainer.Variable(np.asarray([wid], dtype=np.int32)))
            ]
            hy, cy, emb_y_out = model.decoder(hy, cy, emb_y_in)
            wid = np.argmax(F.softmax(model.W(emb_y_out[0])).data[0])
            loop += 1
            if (wid == vocab["<eos>"]) or (loop > 60):
                break
            ys.append(rvocab[wid])
    return ys


def main(mpath):

    # if len(argv) < 6:
    #  print("python "+argv[0]+" x_file vocab rvocab model_path out_path")
    #  sys.exit(0)

    # x_file = argv[1]
    # vocab_path = argv[2]
    # rvocab_path = argv[3]
    # model_path = argv[4]
    # out_path = argv[5]

    if len(argv) < 4:
        print("python " + argv[0] + " data_path model_path out_path")
        sys.exit(0)

    data_path = argv[1]
    model_path = argv[2]
    out_path = argv[3]

    x_file = os.path.join(data_path, "en_test.txt")
    #x_file = os.path.join(data_path, "en_test.txt")
    vocab_path = os.path.join(data_path, "vocab.dump")
    rvocab_path = os.path.join(data_path, "rvocab.dump")

    # 単語とidの辞書
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(rvocab_path, "rb") as f:
        rvocab = pickle.load(f)

    epoch = str(int(model_path.strip(".model").split("-")[2]) + 1)
    layer = int(model_path.strip(".model").split("-")[1])
    out_path = os.path.join(out_path, "seq2seq_" + str(layer) + "_" + epoch + ".txt")

    train_data1 = load_data(x_file, vocab)

    eos_id = vocab["<eos>"]
    demb = 256
    model = seq2seq(layer, len(vocab) + 1, demb, 0.5)
    serializers.load_npz(model_path, model)

    s = []
    ysm = []
    for pos in range(len(train_data1)):
        id = train_data1[pos]
        if id != eos_id:
            s += [id]
        else:
            xs = [np.asarray(s, dtype=np.int32)]
            s = []
            ysm += [" ".join(generate(model, xs, layer, demb, vocab, rvocab))]
            print(ysm[-1])
    with open(out_path, mode="w") as f:
        f.write("\n".join(ysm))


if __name__ == "__main__":

    argv = sys.argv
    main(argv)
