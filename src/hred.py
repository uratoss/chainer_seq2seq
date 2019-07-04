#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import datetime
import sys
import pickle
import numpy as np
import cupy as xp
import chainer
from chainer import cuda, Function, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import chainermn


def load_data(filename, vocab):
    words = open(filename).read().replace("\n", "<eos> ").strip().split()
    dataset = xp.ndarray((len(words),), dtype=xp.int32)
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

    def __call__(self, hx, cx, xs, ys, batch_size, vocab):
        accum_loss = None
        # エンコーダ側の学習
        embs_x = [self.embedx(x) for x in xs]
        hx, cx, __ = self.encoder(hx, cx, embs_x)
        # ctに渡す
        # lay分行う
        # cはcxを使う
        # for hl in hx:
        #  __,cc,yy=self.context(None,None,[hl])
        #  hc_tmp.append(yy[0])
        # hc = F.concat(hc_tmp,axis=0)
        # hc = F.reshape(hc,(hx.data.shape[0],hx.data.shape[1],hx.data.shape[2]))

        # LSTMの入力に単語とctを混ぜるやつ
        cs_in = list(F.split_axis(hx[-1], hx[-1].shape[0], axis=0))
        __, __, cy = self.context(None, None, cs_in)
        ###

        # デコーダ側の学習
        # 出力にはeosを付与
        # ysを入力と教師信号に分ける
        eos = xp.array([vocab["<eos>"]], dtype=xp.int32)
        ys_out = [F.concat((y, eos), axis=0) for y in ys]

        # LSTMの出力にctを混ぜる
        # ys_in = [F.concat((eos, y), axis=0) for y in ys]
        # embs_y_in = [ self.embedy(y) for y in ys_in ]
        # _, _, embs_y_out = self.decoder(hx, cx,embs_y_in )
        ###

        # LSTMの入力に単語とctを混ぜるやつ
        embs_y = [self.embedy(y) for y in ys]
        embs_y_in = []
        for emb_y, h in zip(embs_y, cy):
            y_in_tmp = [
                F.tanh(self.Wc(h) + self.Wy(F.reshape(y_in, (1, y_in.shape[0])))).data
                for y_in in emb_y
            ]
            y_in_tmp.insert(0, h)
            y_in_tmp = F.concat(y_in_tmp, axis=0)
            embs_y_in.append(y_in_tmp)
        _, _, embs_y_out = self.decoder(None, None, embs_y_in)
        ###

        # 損失を計算する
        for emb_y_out, y_out in zip(embs_y_out, ys_out):
            loss = F.softmax_cross_entropy(self.W(emb_y_out), y_out)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        accum_loss = accum_loss / batch_size
        return accum_loss


def main(argv):

    #if len(argv) < 5:
    #    print("python " + argv[0] + " x_file y_file vocab_path out_path")
    #    sys.exit(0)

    #x_file = argv[1]
    #y_file = argv[2]
    #vocab_path = argv[3]
    #out_path = argv[4]

    if len(argv) < 5:
        print("python " + argv[0] + " data_path out_path layer epoch")
        sys.exit(0)

    data_path = argv[1]
    out_path = argv[2]
    layer = int(argv[3])
    epoch = int(argv[4])

    x_file = os.path.join(data_path, "en.txt")
    y_file = os.path.join(data_path, "ja.txt")
    vocab_path = os.path.join(data_path, "vocab.dump")

    # 単語とidの辞書
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    train_data1 = load_data(x_file, vocab)
    train_data2 = load_data(y_file, vocab)

    eos_id = vocab["<eos>"]
    split_id = vocab["<split>"]
    demb = 256
    drop_out = 0.5
    model = hred(layer, len(vocab) + 1, demb, drop_out)

    comm = chainermn.create_communicator("single_node")
    device = comm.intra_rank
    chainer.cuda.get_device(device).use()
    model.to_gpu(device)
    optimizer = chainermn.create_multi_node_optimizer(chainer.optimizers.Adam(), comm)
    optimizer.setup(model)

    # file・directory生成
    date = datetime.datetime.today()
    folder_name = "_".join([str(date.year), str(date.month), str(date.day)])
    out_path = (
        os.path.join(out_path, folder_name, "".join(["layer", str(layer)])) + os.sep
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    loss_out_path = os.path.join(
        out_path, "".join(["loss_", str(epoch), "_", str(layer), ".csv"])
    )
    loss_out = open(loss_out_path, "w")
    print(
        "".join(
            [
                "epoch:",
                str(epoch),
                " drop:",
                str(drop_out),
                " demb:",
                str(demb),
                " layer:",
                str(layer),
            ]
        ),
        end="\n",
        file=loss_out,
    )

    xss = []
    ss = []
    yss = []
    s = []
    if comm.rank == 0:
        # xのデータを生成
        for pos in range(len(train_data1)):
            id = train_data1[pos]
            if id == split_id:
                xss += [ss]
                ss = []
            elif id != eos_id:
                s += [id]
            else:
                if train_data1[pos - 1] == split_id:
                    continue
                ss += [xp.asarray(s, dtype=xp.int32)]
                s = []
        # yのデータを生成
        for pos in range(len(train_data2)):
            id = train_data2[pos]
            if id == split_id:
                yss += [ss]
                ss = []
            elif id != eos_id:
                s += [id]
            else:
                if train_data2[pos - 1] == split_id:
                    continue
                ss += [xp.asarray(s, dtype=xp.int32)]
                s = []
    # データを配る
    xss = chainermn.scatter_dataset(xss, comm)
    yss = chainermn.scatter_dataset(yss, comm)

    loss = None
    for cnt in range(epoch):
        pos = 0
        for xs, ys in zip(xss, yss):
            # 初期値を生成
            model.cleargrads()
            hx = chainer.Variable(xp.zeros((layer, len(xs), demb), dtype=xp.float32))
            cx = chainer.Variable(xp.zeros((layer, len(xs), demb), dtype=xp.float32))
            # 学習する
            loss = model(hx, cx, xs, ys, len(xs), vocab)
            loss.backward()
            optimizer.update()
            pos = pos + 1
            print(cnt, " : ", pos, "/", len(xss), " finished")
        if comm.rank == 0:
            print(loss.array, end="\n", file=loss_out)
            out_file = out_path + "nsteplstm-" + str(layer) + "-" + str(cnt) + ".model"
            model.to_cpu()
            serializers.save_npz(out_file, model)
            model.to_gpu(0)


if __name__ == "__main__":
    argv = sys.argv
    main(argv)
