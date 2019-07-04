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


class hred(chainer.Chain):
    def __init__(self, lay, vocab, k, dout):
        super(hred, self).__init__(
            embedx=L.EmbedID(vocab, k),
            embedy=L.EmbedID(vocab, k),
            encoder=L.NStepLSTM(lay, k, k, dout),
            context=L.NStepLSTM(lay, k, k, dout),
            decoder=L.NStepLSTM(lay, k, k, dout),
            W=L.Linear(k, vocab),
            Wc=L.Linear(k, k),
            Wy=L.Linear(k, k),
        )

    def __call__(self, hx, cx, xs, ys, batch_size):
        accum_loss = None
        # エンコーダ側の学習
        embs_x = [self.embedx(x) for x in xs]
        hx, cx, __ = self.encoder(hx, cx, embs_x)
        # ctに渡す
        # lay分行う
        # cはcxを使う
        cs_in = list(F.split_axis(hx[-1], hx[-1].shape[0], axis=0))
        __, __, hy = self.context(None, None, cs_in)
        # デコーダ側の学習
        # 出力にはeosを付与
        # ysを入力と教師信号に分ける
        eos = xp.array([vocab["<eos>"]], dtype=xp.int32)
        ys_out = [F.concat((y, eos), axis=0) for y in ys]

        embs_y = [self.embedy(y) for y in ys]
        embs_y_in = []
        for emb_y, h in zip(embs_y, hy):
            y_in_tmp = [
                F.tanh(self.Wc(h) + self.Wy(F.reshape(y_in, (1, y_in.shape[0])))).data
                for y_in in emb_y
            ]
            y_in_tmp.insert(0, h)
            y_in_tmp = F.concat(y_in_tmp, axis=0)
            embs_y_in.append(y_in_tmp)
        # _, _, embs_y_out = self.decoder(hx, cx,embs_y_in )
        _, _, embs_y_out = self.decoder(None, None, embs_y_in)

        # 損失を計算する
        for emb_y_out, y_out in zip(embs_y_out, ys_out):
            loss = F.softmax_cross_entropy(self.W(emb_y_out), y_out)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        accum_loss = accum_loss / batch_size
        return accum_loss


def generate(model, xs, layer, demb, hc, cc, vocab, rvocab):
    with chainer.using_config("enable_backprop", False), chainer.using_config(
        "train", False
    ):
        # 初期値を生成
        hx = chainer.Variable(np.zeros((layer, len(xs), demb), dtype=np.float32))
        cx = chainer.Variable(np.zeros((layer, len(xs), demb), dtype=np.float32))
        # エンコーダ側の処理
        embs_x = [model.embedx(x) for x in xs]
        hx, cx, _ = model.encoder(hx, cx, embs_x)
        # ctに渡す
        hc, cc, yy = model.context(hc, cc, [hx[-1]])
        hy = None
        cy = None
        # デコーダ側の処理
        # eosを入力して始める
        ys = []
        loop = 0
        hy, cy, emb_y_out = model.decoder(hy, cy, yy)
        wid = np.argmax(F.softmax(model.W(emb_y_out[0])).data[0])
        ys.append(rvocab[wid])
        while True:
            emb_y_in = model.embedy(chainer.Variable(np.asarray([wid], dtype=np.int32)))
            y_in = F.tanh(model.Wc(yy[0]) + model.Wy(emb_y_in))
            hy, cy, emb_y_out = model.decoder(hy, cy, [y_in])
            wid = np.argmax(F.softmax(model.W(emb_y_out[0])).data[0])
            loop += 1
            if (wid == vocab["<eos>"]) or (loop > 30):
                break
            ys.append(rvocab[wid])
    return ys, hc, cc


def main(mpath):

    #if len(argv) < 6:
    #    print("python " + argv[0] + " x_file vocab rvocab model_path out_path")
    #    sys.exit(0)

    #x_file = argv[1]
    #vocab_path = argv[2]
    #rvocab_path = argv[3]
    #model_path = argv[4]
    #out_path = argv[5]

    if len(argv) < 4:
        print("python " + argv[0] + " data_path model_path out_path")
        sys.exit(0)

    data_path = argv[1]
    model_path = argv[2]
    out_path = argv[3]

    x_file = os.path.join(data_path, "en_test.txt")
    vocab_path = os.path.join(data_path, "vocab.dump")
    rvocab_path = os.path.join(data_path, "rvocab.dump")

    # 単語とidの辞書
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(rvocab_path, "rb") as f:
        rvocab = pickle.load(f)

    epoch = str(int(model_path.strip(".model").split("-")[2]) + 1)
    layer = int(model_path.strip(".model").split("-")[1])
    out_path = os.path.join(out_path, "hred_" + str(layer) + "_" + epoch + ".txt")

    train_data1 = load_data(x_file, vocab)

    eos_id = vocab["<eos>"]
    split_id = vocab["<split>"]
    demb = 256
    model = hred(layer, len(vocab) + 1, demb, 0.25)
    serializers.load_npz(model_path, model)

    s = []
    ysm = []
    hc = None
    cc = None
    for pos in range(len(train_data1)):
        id = train_data1[pos]
        if id == split_id:
            hc = None
            cc = None
            ysm += ["<split>"]
            continue
        elif id != eos_id:
            s += [id]
        else:
            if train_data1[pos - 1] == split_id:
                continue
            xs = [np.asarray(s, dtype=np.int32)]
            s = []
            seq, hc, cc = generate(model, xs, layer, demb, hc, cc, vocab, rvocab)
            ysm += [" ".join(seq)]
            print(ysm[-1])
    with open(out_path, mode="w") as f:
        f.write("\n".join(ysm))


if __name__ == "__main__":
    argv = sys.argv
    main(argv)
