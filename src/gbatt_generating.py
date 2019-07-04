#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import math
import numpy as xp
import chainer
from chainer import cuda, Function, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


def load_data(filename, vocab):
    words = open(filename).read().replace("\n", "<eos> ").strip().split()
    dataset = xp.ndarray((len(words),), dtype=xp.int32)
    for i, word in enumerate(words):
        dataset[i] = vocab[word]
    return dataset


# 'gAtt'{{{
class gAtt(chainer.Chain):
    def __init__(self, lay, vocab, k, dout):
        super(gAtt, self).__init__(
            embedx=L.EmbedID(vocab, k),
            embedy=L.EmbedID(vocab, k),
            encoder=L.NStepBiLSTM(lay, k, k, dout),
            decoder=L.NStepLSTM(lay, k, k, dout),
            W=L.Linear(k, vocab),
            W1=L.Linear(2 * k, k),
            W2=L.Linear(2 * k, k),
            W3=L.Linear(2 * k, k),
            Wa1=L.Linear(k, k),
            Wa2=L.Linear(k, k),
            Wa3=L.Linear(k, k),
            Wc1=L.Linear(k, k),
            Wc2=L.Linear(k, k),
            Wc3=L.Linear(k, k),
        )

    # 'dotScore' {{{
    def scoreDot(self, attentions, dec_ys_out):
        emb_ys_out = []
        for attention, dec_y_out in zip(attentions, dec_ys_out):
            # attention -> 1文のエンコーダの出力
            # dec_y_out -> 1文のデコーダの出力
            attention = self.W3(attention)
            target = []
            for dec_y in dec_y_out:
                dec_y = F.reshape(dec_y, (1, dec_y.shape[0]))
                # dec_y -> デコーダLSTMブロックの1単語の出力
                # ctを計算する
                s = 0.0
                scores = []
                for enc in attention:
                    enc = F.reshape(enc, (1, enc.shape[0]))
                    # ecn -> エンコーダLSTMブロックの1単語の出力
                    scores.append(F.exp(F.matmul(dec_y, enc, transb=True)).data[0][0])
                    s += scores[-1]
                ct = xp.zeros((1, dec_y.shape[-1]), dtype=xp.float32)
                for enc, score in zip(attention, scores):
                    # ecn -> エンコーダLSTMブロックの1単語の出力
                    # score -> 各エンコーダの出力に対するdot積の結果
                    alpi = score / s
                    ct += alpi * enc.data[0]
                # ctから出力単語を作る
                ct = chainer.Variable(ct)
                target.append(F.tanh(self.Wc1(ct) + self.Wc2(dec_y)))
            emb_ys_out.append(F.concat(target, axis=0))
        return emb_ys_out

    # }}}

    # 'generalScore' {{{
    def scoreGeneral(self, attentions, dec_ys_out):
        emb_ys_out = []
        for attention, dec_y_out in zip(attentions, dec_ys_out):
            # attention -> 1文のエンコーダの出力
            # dec_y_out -> 1文のデコーダの出力

            # エンコーダの出力を順方向・逆方向に分ける
            attention_f, attention_b = F.split_axis(attention, 2, axis=1)
            target = []
            for dec_y in dec_y_out:
                dec_y = F.reshape(dec_y, (1, dec_y.shape[0]))
                # dec_y -> デコーダLSTMブロックの1単語の出力
                # ctを計算する
                s = 0.0
                scores = []
                for enc_f, enc_b in zip(attention_f, attention_b):
                    enc_f = F.reshape(enc_f, (1, enc_f.shape[0]))
                    enc_b = F.reshape(enc_b, (1, enc_b.shape[0]))
                    # ecn_f -> 順方向エンコーダLSTMブロックの1単語の出力
                    # ecn_b -> 逆方向エンコーダLSTMブロックの1単語の出力
                    scores.append(
                        F.exp(self.Wa1(enc_f) + self.Wa2(enc_b) + self.Wa3(dec_y)).data[
                            0
                        ][0]
                    )
                    s += scores[-1]
                ct_f = xp.zeros((1, dec_y.shape[-1]), dtype=xp.float32)
                ct_b = xp.zeros((1, dec_y.shape[-1]), dtype=xp.float32)
                for enc_f, enc_b, score in zip(attention_f, attention_b, scores):
                    enc_f = F.reshape(enc_f, (1, enc_f.shape[0]))
                    enc_b = F.reshape(enc_b, (1, enc_b.shape[0]))
                    # ecn_f -> 順方向エンコーダLSTMブロックの1単語の出力
                    # ecn_b -> 逆方向エンコーダLSTMブロックの1単語の出力
                    alpi = score / s
                    ct_f += alpi * enc_f.data[0]
                    ct_b += alpi * enc_b.data[0]
                # ctから出力単語を作る
                ct_f = chainer.Variable(ct_f)
                ct_b = chainer.Variable(ct_b)
                target.append(F.tanh(self.Wc1(ct_f) + self.Wc2(ct_b) + self.Wc3(dec_y)))
            emb_ys_out.append(F.concat(target, axis=0))
        return emb_ys_out

    # }}}

    def __call__(self, hx, cx, xs, ys, batch_size, vocab):
        accum_loss = None
        # エンコーダ側の学習
        embs_x = [self.embedx(x) for x in xs]
        hx, cx, attentions = self.encoder(hx, cx, embs_x)

        # 双方向なので、デコーダ用に次元を合わせる
        # hx(lay*2,batch,demb) -> hx(lay,batch,demb)
        hx_lays = F.reshape(
            hx, (int(hx.shape[0] / 2), int(hx.shape[1] * 2), int(hx.shape[2]))
        )
        hx_lays_tmp = []
        for hx_lay in hx_lays:
            hx_list = list(F.split_axis(hx_lay, hx_lay.shape[0], axis=0))
            hx_list_tmp = []
            for i in range(int(len(hx_list) / 2)):
                hx_list_tmp.append(
                    F.concat((hx_list[i], hx_list[i + int(len(hx_list) / 2)]), axis=1)
                )
            hx_lays_tmp.append(F.concat(hx_list_tmp, axis=0))
        hx = self.W1(F.concat(hx_lays_tmp, axis=0))
        hx = F.reshape(
            hx, (int(hx.shape[0] / batch_size), batch_size, int(hx.shape[1]))
        )
        cx_lays = F.reshape(
            cx, (int(cx.shape[0] / 2), int(cx.shape[1] * 2), int(cx.shape[2]))
        )
        cx_lays_tmp = []
        for cx_lay in cx_lays:
            cx_list = list(F.split_axis(cx_lay, cx_lay.shape[0], axis=0))
            cx_list_tmp = []
            for i in range(int(len(cx_list) / 2)):
                cx_list_tmp.append(
                    F.concat((cx_list[i], cx_list[i + int(len(cx_list) / 2)]), axis=1)
                )
            cx_lays_tmp.append(F.concat(cx_list_tmp, axis=0))
        cx = self.W2(F.concat(cx_lays_tmp, axis=0))
        cx = F.reshape(
            cx, (int(cx.shape[0] / batch_size), batch_size, int(cx.shape[1]))
        )

        # デコーダ側の学習
        # eosを付与
        # ysを入力と教師信号に分ける
        eos = xp.array([vocab["<eos>"]], dtype=xp.int32)
        ys_in = [F.concat((eos, y), axis=0) for y in ys]
        ys_out = [F.concat((y, eos), axis=0) for y in ys]

        embs_y_in = [self.embedy(y) for y in ys_in]
        _, _, dec_ys_out = self.decoder(hx, cx, embs_y_in)
        # attを計算する
        emb_ys_out = self.scoreDot(attentions, dec_ys_out)
        # emb_ys_out = self.scoreGeneral(attentions, dec_ys_out)

        # 損失を計算する
        for emb_y_out, y_out in zip(emb_ys_out, ys_out):
            loss = F.softmax_cross_entropy(self.W(emb_y_out), y_out)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        accum_loss = accum_loss / batch_size
        return accum_loss


# }}}


def generate(model, xs, layer, demb, vocab, rvocab, batch_size):
    with chainer.using_config("enable_backprop", False), chainer.using_config(
        "train", False
    ):
        # 初期値を生成
        hx = chainer.Variable(xp.zeros((2 * layer, len(xs), demb), dtype=xp.float32))
        cx = chainer.Variable(xp.zeros((2 * layer, len(xs), demb), dtype=xp.float32))
        # エンコーダ側の処理
        embs_x = [model.embedx(x) for x in xs]
        hx, cx, attentions = model.encoder(hx, cx, embs_x)

        # 双方向なので、デコーダ用に次元を合わせる
        # hx(lay*2,batch,demb) -> hx(lay,batch,demb)
        hx_lays = F.reshape(
            hx, (int(hx.shape[0] / 2), int(hx.shape[1] * 2), int(hx.shape[2]))
        )
        hx_lays_tmp = []
        for hx_lay in hx_lays:
            hx_list = list(F.split_axis(hx_lay, hx_lay.shape[0], axis=0))
            hx_list_tmp = []
            for i in range(int(len(hx_list) / 2)):
                hx_list_tmp.append(
                    F.concat((hx_list[i], hx_list[i + int(len(hx_list) / 2)]), axis=1)
                )
            hx_lays_tmp.append(F.concat(hx_list_tmp, axis=0))
        hx = model.W1(F.concat(hx_lays_tmp, axis=0))
        hx = F.reshape(
            hx, (int(hx.shape[0] / batch_size), batch_size, int(hx.shape[1]))
        )
        cx_lays = F.reshape(
            cx, (int(cx.shape[0] / 2), int(cx.shape[1] * 2), int(cx.shape[2]))
        )
        cx_lays_tmp = []
        for cx_lay in cx_lays:
            cx_list = list(F.split_axis(cx_lay, cx_lay.shape[0], axis=0))
            cx_list_tmp = []
            for i in range(int(len(cx_list) / 2)):
                cx_list_tmp.append(
                    F.concat((cx_list[i], cx_list[i + int(len(cx_list) / 2)]), axis=1)
                )
            cx_lays_tmp.append(F.concat(cx_list_tmp, axis=0))
        cx = model.W2(F.concat(cx_lays_tmp, axis=0))
        cx = F.reshape(
            cx, (int(cx.shape[0] / batch_size), batch_size, int(cx.shape[1]))
        )

        # デコーダ側の処理
        # eosが出力されたら終わり
        wid = vocab["<eos>"]
        ys = []
        loop = 0
        while True:
            emb_y_in = [
                model.embedy(chainer.Variable(xp.asarray([wid], dtype=xp.int32)))
            ]
            hx, cx, dec_y_out = model.decoder(hx, cx, emb_y_in)
            # attを計算する
            emb_ys_out = model.scoreDot(attentions, dec_y_out)
            wid = xp.argmax(F.softmax(model.W(emb_ys_out[0])).data[0])
            loop += 1
            if (wid == vocab["<eos>"]) or (loop > 60):
                break
            ys.append(rvocab[wid])
    return ys


def main(mpath):
    # if len(argv) < 6:
    #    print("python " + argv[0] + " x_file vocab rvocab model_path out_path")
    #    sys.exit(0)

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

    #x_file = os.path.join(data_path, "en_develop.txt")
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
    out_path = os.path.join(out_path, "attention_" + str(layer) + "_" + epoch + ".txt")

    train_data1 = load_data(x_file, vocab)

    eos_id = vocab["<eos>"]
    demb = 256
    model = gAtt(layer, len(vocab) + 1, demb, 0.5)
    serializers.load_npz(model_path, model)

    s = []
    ysm = []
    for pos in range(len(train_data1)):
        id = train_data1[pos]
        if id != eos_id:
            s += [id]
        else:
            xs = [xp.asarray(s, dtype=xp.int32)]
            s = []
            ysm += [" ".join(generate(model, xs, layer, demb, vocab, rvocab, len(xs)))]
            print(ysm[-1])
    with open(out_path, mode="w") as f:
        f.write("\n".join(ysm))


if __name__ == "__main__":

    argv = sys.argv
    main(argv)
