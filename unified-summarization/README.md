# Unified Summarization

This is the official codes for the paper: [A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss](https://arxiv.org/abs/1805.06266).

## Requirements

* Python 2.7
* [Tensoflow 1.1.0](https://www.tensorflow.org/versions/r1.1/)
* [pyrouge](https://pypi.org/project/pyrouge/) (for evaluation)
* tqdm
* [Standford CoreNLP 3.7.0](https://stanfordnlp.github.io/CoreNLP/) (for data preprocessing)
* [NLTK](https://www.nltk.org/) (for data preprocessing)


**Note**: Stanford CoreNLP 3.7.0 can be downloaded from [here](http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip).

**Note**: To use ROUGE evaluation, you need to download the `ROUGE-1.5.5` package from [here](https://github.com/andersjo/pyrouge). Next, follow the instrunction from [here](https://pypi.org/project/pyrouge/) to install pyrouge and set the ROUGE path to your absolute path of `ROUGE-1.5.5` directory.

**Error Handling**: If you encounter the error message `Cannot open exception db file for reading: /path/to/ROUGE-1.5.5/data/WordNet-2.0.exc.db` when using pyrouge, the problem can be solved from [here](https://github.com/tagucci/pythonrouge#error-handling).

## CNN/Daily Mail dataset

Codes for generating the dataset is in `data` folder.

We modified the preprocessing code from [this repository](https://github.com/abisee/cnn-dailymail).

You can use our preprocessing codes ([data/make_datafiles.py](./data/make_datafiles.py) and [data/rouge_not_a_wrapper.py](./data/rouge_not_a_wrapper.py)) and follow their instrunctions of [Option 2](https://github.com/abisee/cnn-dailymail#option-2-process-the-data-yourself) to obtain the preprocessed data for our model.


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

Set the path of pretrained extractor and abstractor to `SELECTOR_PATH` and `REWRITER_PATH` in the script.

```
sh scripts/end2end.sh
```

The trained models will be saved in `log/end2end/${YOUR_EXP_NAME}` directory.

**Note**: In our paper, we use the best extractor model on validation set for the pretrained extractor and the last abstracter model (after training with coverage mechanism for 1k iterations) for the pretrained abstracter in end-to-end training.

## How to evaluate (concurrent)

To evaluate the model during training, change the `MODE` in the script to `eval` (i.e., `MODE='eval'`) and run the script simutanously with train script (i.e., `MODE='train'`).

For evaluating the abstracter and the unified model, you can choose to evaluate the loss or ROUGE scores. Just switch the `EVAL_METHOD` in the script between `loss` and `rouge`. 

For the ROUGE evaluation, you can use greedy search or beam search. Just switch the `DECODE_METHOD` in the script between `greedy` and `beam`.

We highly recommend you to use **greedy search** for concorrent ROUGE evaluation since greedy search is much faster than beam search.
It takes about 30 minutes for greedy search and 7 hours for beam search on CNN/Daily Mail test set.

The current best models will be saved in `log/${MODEL}/${YOUR_EXP_NAME}/eval_${DATA_SPLIT}(_${EVAL_METHOD})`.

## How to evaluate with ROUGE on test set

Change the `MODE` in the script to `evalall` (i.e., `MODE='evalall'`) and set `CKPT_PATH` as the model path that you want to test.

If you want to use the best evaluation model, set `LOAD_BEST_VAL_MODEL` or `LOAD_BEST_TEST_MODEL` as `True` to load the best model in `eval_val(_${EVAL_METHOD})` or `eval_test(_${EVAL_METHOD})` directory. The default of `LOAD_BEST_VAL_MODEL` and `LOAD_BEST_TEST_MODEL` are `False`.

If you didn't set the `CKPT_PATH` or turn on `LOAD_BEST_VAL(TEST)_MODEL`, it will automatically load the latest model in `train` directory.

The evalutation results will be saved under your experiment directory `log/${MODEL}/${YOUR_EXP_NAME}/`.

## Expected Results

By following the scripts we provided, you should get the performance as below:

### Extractor (best evaluation model)

| ROUGE-1 recall | ROUGE-1 recall |ROUGE-1 recall| 
|:----------:|:---------:|:-----------:|
|   73.5     |    35.6   |   68.6    |

### Abstracter (model of 81000 iteration)

| ROUGE-1 F-1 score | ROUGE-1 F-1 score |ROUGE-1 F-1 score| 
|:----------:|:---------:|:-----------:|
|     45.4   |    21.8   |   42.1   |

### Unified model with inconsistency loss (best evaluation model)

| ROUGE-1 F-1 score | ROUGE-1 F-1 score |ROUGE-1 F-1 score| 
|:----------:|:---------:|:-----------:|
|     40.68   |    17.97   |   37.13   |


**Note**: Our abstracter takes ground-truth extracted sentences as input when both training and testing, so the ROUGE F-1 scores are higher than the unified model.


## Our test set outputs

Test set outputs of our unified model can be download from [here](https://hsuwanting.github.io/unified_summ/unified_test_output.zip).

Each pickle file (e.g., `result_000000.pkl`) contains the output of one article.

The output format is a dictionary:

```
{
    'article': list of article sentences,
    'reference': list of reference summary sentences,
    'gt_ids': indices of ground-truth extracted sentences,
    'decoded': list of output summary sentences
}
```

