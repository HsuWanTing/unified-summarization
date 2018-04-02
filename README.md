# Unified Summarization

## CNN/Daily mail dataset

Codes for generating the dataset is in `data_preprocess` folder.

The usage is same as https://github.com/abisee/cnn-dailymail


## How to train

Use the sample scripts in `scripts` folder.

### Pretrain the exatrctor

```
sh scripts/selector.sh
```

### Pretrain the abstracter

```
sh scripts/rewriter.sh
```

### End-to-end training

```
sh scripts/end2end.sh
```

## How to evaluate (concurrent)

To evaluate the model during training, change the `MODE` in the script to `eval` (i.e., `MODE='eval'`) and run the script simutanously with train script (i.e., `MODE='train'`).

For evaluating the abstracter and the end2end model, you can choose to evaluate the loss or ROUGE scores. Just switch the `EVAL_METHOD` in the script to `loss` or `rouge`. 

For the ROUGE evaluation, you can use greedy search or beam search. Just switch the `DECODE_METHOD` in the script to `greedy` or `beam`.

## How to evaluate with ROUGE for test set

Change the `MODE` in the script to `evalall` (i.e., `MODE='evalall'`) and set the `CKPT_PATH` as the model path that you want to test. 
