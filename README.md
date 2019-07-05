chainer_seq2seq
====

seq2seq using chainer

## Description
``` 
(root)  
|  
+--data : data-set
|
+--test : Output location of test
|
+--model : Trained models
|   |
|   +--attention : seq2seq model with global attention structure
|   |
|   +--hred : hred model
|   |
|   L--nstep : seq2seq with nstep lstm
|
L--src : some learning and generating codes
    |
    L--tools : some tools, e.g. to make vocabulary data
```
In src directory,details are given as under.
 - hogehoge_generating.py
   - output sequence using models was trained by hogehoge
 - nsteplstm.py
   - normal seq2seq using nsteplstm
 - gbatt.py
   - seq2seq using nsteplstm with global attention structure
 - hred.py
   - hred model
   
## Usage
1. install `mecab, numpy, chainer, chainermn, cupy, MPI, NCCL, CUDA, and so on` for your machine 
 - sorry if not enough
2. You have to place data-set for data directry  
Name input data as `en`,and name trainig data as `ja`
```
(data)
|
+--{en,ja}.txt : en is input data, ja is training data.
|
+--{en,ja}_devel.txt : data using adjust for model's parameter 
|
+--{en,ja}_test.txt : data using test for model
```
3. Using `tools/mkvocab.py`,you make vocabulary data of your data-set
4. Starting learn by run learning codes
