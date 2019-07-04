#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import os
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
        hx, cx, _ = self.encoder(hx, cx, embs_x)
        # デコーダ側の学習
        # eosを付与
        # ysを入力と教師信号に分ける
        eos = xp.array([vocab["<eos>"]], dtype=xp.int32)
        ys_in = [F.concat((eos, y), axis=0) for y in ys]
        ys_out = [F.concat((y, eos), axis=0) for y in ys]

        embs_y_in = [self.embedy(y) for y in ys_in]
        _, _, embs_y_out = self.decoder(hx, cx, embs_y_in)

        # print(y_out.shape)
        # 損失を計算する
        for emb_y_out, y_out in zip(embs_y_out, ys_out):
            loss = F.softmax_cross_entropy(self.W(emb_y_out), y_out)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        accum_loss = accum_loss / batch_size
        return accum_loss


def main(argv):

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
    batch_size = 256
    demb = 256
    drop_out = 0.5
    model = seq2seq(layer, len(vocab) + 1, demb, drop_out)

    comm = chainermn.create_communicator("single_node")
    device = comm.intra_rank
    chainer.cuda.get_device(device).use()
    model.to_gpu(device)
    optimizer = chainermn.create_multi_node_optimizer(chainer.optimizers.Adam(), comm)
    optimizer.setup(model)

    if comm.rank == 0:
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
                    " batch:",
                    str(batch_size),
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

    xs = []
    ys = []
    s = []
    # xのデータを生成
    if comm.rank == 0:
        for pos in range(len(train_data1)):
            id = train_data1[pos]
            if id != eos_id:
                s += [id]
            else:
                xs += [xp.asarray(s, dtype=xp.int32)]
                s = []
        # yのデータを生成
        for pos in range(len(train_data2)):
            id = train_data2[pos]
            if id != eos_id:
                s += [id]
            else:
                ys += [xp.asarray(s, dtype=xp.int32)]
                s = []
    # データを配る
    xs = chainermn.scatter_dataset(xs, comm)
    ys = chainermn.scatter_dataset(ys, comm)

    loss = None
    for cnt in range(epoch):
        index = np.random.permutation(len(xs))
        for pos in range(0, len(xs), batch_size):
            # ミニバッチを生成
            batch_xs = []
            batch_ys = []
            for idx in index[pos : pos + (batch_size)]:
                batch_xs.append(xs[idx])
                batch_ys.append(ys[idx])
            model.cleargrads()
            # 初期値を生成
            hx = chainer.Variable(
                xp.zeros((layer, len(batch_xs), demb), dtype=xp.float32)
            )
            cx = chainer.Variable(
                xp.zeros((layer, len(batch_xs), demb), dtype=xp.float32)
            )
            # 学習する
            loss = model(hx, cx, batch_xs, batch_ys, len(batch_xs), vocab)
            loss.backward()
            optimizer.update()
            print(cnt + 1, " : ", pos + len(batch_xs), "/", len(xs), " finished")
        if comm.rank == 0:
            print(str(loss.array), end="\n", file=loss_out)
            out_file = out_path + "nsteplstm-" + str(layer) + "-" + str(cnt) + ".model"
            model.to_cpu()
            serializers.save_npz(out_file, model)
            model.to_gpu(0)


if __name__ == "__main__":
    argv = sys.argv
    main(argv)
