# Unified Summarization

This is the official codes for the paper: [A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss](https://arxiv.org/abs/1805.06266).

## Requirements

* Python 2.7
* [Tensoflow 1.1.0](https://www.tensorflow.org/versions/r1.1/)
* [Standford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) (for data preprocessing)
* [NLTK](https://www.nltk.org/) (for data preprocessing)
* [pyrouge](https://pypi.org/project/pyrouge/) (for evaluation)
* matplotlib
* tqdm

## CNN/Daily Mail dataset

Codes for generating the dataset is in `data` folder.

We modified the preprocessing code from [this repository](https://github.com/abisee/cnn-dailymail).

You can use our preprocessing codes ([data/make_datafiles.py](./data/make_datafiles.py) and [data/rouge_not_a_wrapper.py](./data/rouge_not_a_wrapper.py)) and follow their instrunctions of [Option 2](https://github.com/abisee/cnn-dailymail#option-2-process-the-data-yourself) to obtain the preprocessed data for our model.

**Note**: Stanford CoreNLP 3.7.0 can be downloaded from [here](http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip). 


## How to train

Use the sample scripts in `scripts` folder. 

### Pretrain the exatrctor

```
sh scripts/selector.sh
```
The trained models will be saved in `log/selector/${YOUR_EXP_NAME}` directory.

### Pretrain the abstracter

```
sh scripts/rewriter.sh
```
The trained models will be saved in `log/rewriter/${YOUR_EXP_NAME}` directory.

### End-to-end training the unified model

Set the path of pretrained extractor and abstractor to `SELECTOR_PATH` and `REWRITER_PATH` in line 18 and 19.

```
sh scripts/end2end.sh
```

The trained models will be saved in `log/end2end/${YOUR_EXP_NAME}` directory.

## How to evaluate (concurrent)

To evaluate the model during training, change the `MODE` in the script to `eval` (i.e., `MODE='eval'`) and run the script simutanously with train script (i.e., `MODE='train'`).

For evaluating the abstracter and the unified model, you can choose to evaluate the loss or ROUGE scores. Just switch the `EVAL_METHOD` in the script between `loss` and `rouge`. 

For the ROUGE evaluation, you can use greedy search or beam search. Just switch the `DECODE_METHOD` in the script between `greedy` and `beam`.

We highly recommend you to use **greedy search** for concorrent ROUGE evaluation since greedy search is much faster than beam search.
It takes about 30 minutes for greedy search and 7 hours for beam search on CNN/Daily Mail test set.

The current best models will be saved in `log/${MODEL}/${YOUR_EXP_NAME}/eval_${DATA_SPLIT}(_${EVAL_METHOD})`.

## How to evaluate with ROUGE on test set

Change the `MODE` in the script to `evalall` (i.e., `MODE='evalall'`) and set the `CKPT_PATH` as the model path that you want to test. 
