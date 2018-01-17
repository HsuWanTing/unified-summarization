# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ROUGE Metric Implementation
This is a very slightly version of:
https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
---
ROUGe metric implementation.
This is a modified and slightly extended verison of
https://github.com/miso-belica/sumy/blob/dev/sumy/evaluation/rouge.py.
"""
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import itertools
import numpy as np
import re
from collections import defaultdict
import pdb

#pylint: disable=C0103


def _get_ngrams(n, text):
  """Calcualtes n-grams.
  Args:
    n: which n-grams to calculate
    text: An array of tokens
  Returns:
    A set of n-grams
  """
  ngram_dict = defaultdict(int)
  text_length = len(text)
  max_index_ngram_start = text_length - n + 1
  for i in range(max_index_ngram_start):
    if n > 1:
      ngram_dict[tuple(text[i:i + n])] += 1
    else:
      ngram_dict[text[i]] += 1
  return ngram_dict, max_index_ngram_start

def _preprocess(sentence):
  """preprocess one sentence (a single string)"""
  #s = sentence.decode('utf-8')
  s = sentence.lower()
  s = re.sub('-', ' - ', s.decode('utf-8'))
  #s = re.sub('-', ' - ', s)
  s = re.sub('[^A-Za-z0-9\-]', ' ', s) # replace not A~Z, a~z, 0~9 to a space
  s = s.strip()
  return s

def _split_into_words(sentences):
  """Splits multiple sentences into words and flattens the result.
     sentences: list of strings"""

  return list(itertools.chain(*[_preprocess(s).split() for s in sentences]))


def _get_word_ngrams(n, sentences):
  """Calculates word n-grams for multiple sentences.
  """
  assert len(sentences) > 0
  assert n > 0

  words = _split_into_words(sentences)
  return _get_ngrams(n, words)


def _len_lcs(x, y):
  """
  Returns the length of the Longest Common Subsequence between sequences x
  and y.
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
  Args:
    x: sequence of words
    y: sequence of words
  Returns
    integer: Length of LCS between x and y
  """
  table = _lcs(x, y)
  n, m = len(x), len(y)
  return table[n, m]


def _lcs(x, y):
  """
  Computes the length of the longest common subsequence (lcs) between two
  strings. The implementation below uses a DP programming algorithm and runs
  in O(nm) time where n = len(x) and m = len(y).
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
  Args:
    x: collection of words
    y: collection of words
  Returns:
    Table of dictionary of coord and len lcs
  """
  n, m = len(x), len(y)
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 or j == 0:
        table[i, j] = 0
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  return table


def _recon_lcs(x, y):
  """
  Returns the Longest Subsequence between x and y.
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
  Args:
    x: sequence of words, a reference sentence
    y: sequence of words, a evaluated sentence
  Returns:
    sequence: LCS of x and y, 
              a list of tuples, 
              each tuple indicates (hit unigram, unigram index in x)
  """
  i, j = len(x), len(y)
  table = _lcs(x, y)
  if table[i, j] == 0:
    return []
  
  lcs = []
  while 1:
    if i == 0 or j == 0:
      break
    elif x[i - 1] == y[j - 1]:
      lcs = [(x[i - 1], i - 1)] + lcs
      i = i - 1
      j = j - 1
    elif table[i - 1, j] > table[i, j - 1]:
      i = i - 1
    else:
      j = j - 1

  '''
  def _recon(i, j):
    """private recon calculation"""
    if i == 0 or j == 0:
      return []
    elif x[i - 1] == y[j - 1]:
      return _recon(i - 1, j - 1) + [(x[i - 1], i - 1)]
    elif table[i - 1, j] > table[i, j - 1]:
      return _recon(i - 1, j)
    else:
      return _recon(i, j - 1)

  LCS = _recon(len(x), len(y))
  pdb.set_trace()
  '''
  return lcs


def rouge_n(evaluated_sentences, reference_sentences, n=2):
  """
  Computes ROUGE-N of two text collections of sentences.
  Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
  papers/rouge-working-note-v1.3.1.pdf
  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set
    n: Size of ngram.  Defaults to 2.
  Returns:
    A tuple (f1, precision, recall) for ROUGE-N
  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    #raise ValueError("Collections must contain at least 1 sentence.")
    return 0.0, 0.0, 0.0

  evaluated_ngrams, evaluated_count = _get_word_ngrams(n, evaluated_sentences)
  reference_ngrams, reference_count = _get_word_ngrams(n, reference_sentences)

  # Gets the overlapping ngrams between evaluated and reference
  overlapping_count = 0
  for ngram in reference_ngrams:
    if ngram in evaluated_ngrams:
      count1 = reference_ngrams[ngram]
      count2 = evaluated_ngrams[ngram]
      hit = count1 if count1 < count2 else count2
      overlapping_count += hit

  return _f_p_r_1(overlapping_count, reference_count, evaluated_count)


def _f_p_r_1(l, m, n):
  """
  Computes the F-measure score
  Args:
    l: overlapping count
    m: number of words in reference summary
    n: number of words in candidate summary
  Returns:
    Float. F-measure score, Precision score, Recall score
  """
  r = l / m if m > 0 else 0.0
  p = l / n if n > 0 else 0.0

  if r + p == 0:
    f = 0.0
  else:
    f = 2.0 * ((r * p) / (r + p))
  return f, p, r


def _f_p_r_2(l, m, n):
  """
  Computes the LCS-based F-measure score
  Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf
  Args:
    l: overlapping count
    m: number of words in reference summary
    n: number of words in candidate summary
  Returns:
    Float. LCS-based F-measure score
  """
  r = l / m if m > 0 else 0.0
  p = l / n if n > 0 else 0.0
  
  beta = p / (r + 1e-12)
  num = (1 + (beta**2)) * r * p
  denom = r + ((beta**2) * p)
  f =  num / (denom + 1e-12)
  return f, p, r

'''
def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
  """
  Computes ROUGE-L (sentence level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf
  Calculated according to:
  R_lcs = LCS(X,Y)/m
  P_lcs = LCS(X,Y)/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
  where:
  X = reference summary
  Y = Candidate summary
  m = length of reference summary
  n = length of candidate summary
  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set
  Returns:
    A float: F_lcs
  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")
  reference_words = _split_into_words(reference_sentences)
  evaluated_words = _split_into_words(evaluated_sentences)
  m = len(reference_words)
  n = len(evaluated_words)
  lcs = _len_lcs(evaluated_words, reference_words)
  return _f_p_r_1(lcs, m, n)
'''

def _union_lcs(evaluated_sentences, reference_sentence):
  """
  Returns LCS_u(r_i, C) which is the LCS score of the union longest common
  subsequence between reference sentence ri and candidate summary C. For example
  if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8 and
  c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1 is
  “w1 w2” and the longest common subsequence of r_i and c2 is “w1 w3 w5”. The
  union longest common subsequence of r_i, c1, and c2 is “w1 w2 w3 w5” and
  LCS_u(r_i, C) = 4.
  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentence: One of the sentences in the reference summaries
  Returns:
    a list of tuples, each tuple indicates (hit_unigram, unigram index in reference)
  ValueError:
    Raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0:
    return set()
    #raise ValueError("Collections must contain at least 1 sentence.")

  lcs_union = set()
  reference_words = _split_into_words([reference_sentence])
  combined_lcs_length = 0
  for eval_s in evaluated_sentences:
    evaluated_words = _split_into_words([eval_s])
    lcs = set(_recon_lcs(reference_words, evaluated_words))
    lcs_union = lcs_union.union(lcs) # a list of tuple (hit_unigram, index in reference)

  return lcs_union


def rouge_l_summary_level(evaluated_sentences, reference_sentences):
  """
  Computes ROUGE-L (summary level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf
  Calculated according to:
  R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
  P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
  where:
  SUM(i,u) = SUM from i through u
  u = number of sentences in reference summary
  C = Candidate summary made up of v sentences
  m = number of words in reference summary
  n = number of words in candidate summary
  Args:
    evaluated_sentences: list of sentences string
    reference_sentence: list of sentences string
  Returns:
    3 float: F-measure score, Precision score, Recall score
  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    return 0.0, 0.0, 0.0
    #raise ValueError("Collections must contain at least 1 sentence.")

  # unigram dictionary for reference and evaluated sentences
  ref_1gram_dict, m = _get_word_ngrams(1, reference_sentences)
  eval_1gram_dict, n = _get_word_ngrams(1, evaluated_sentences)

  total_hits = 0
  for ref_s in reference_sentences:
    ref_hits = list(_union_lcs(evaluated_sentences, ref_s))
    for w in ref_hits:
      # bookkeeping to clip over counting everytime a hit is found,
      # it is deducted from both ref and eval unigram count.
	  # If a unigram count already involve in one LCS match 
      # then it will not be counted if it match another token in the ref unit. 
      # This will make sure LCS score is always lower than unigram score
      if ref_1gram_dict[w[0]] > 0 and eval_1gram_dict[w[0]] > 0:
        total_hits += 1
        ref_1gram_dict[w[0]] -= 1
        eval_1gram_dict[w[0]] -= 1
  return _f_p_r_1(total_hits, m, n)

'''
def rouge(hypotheses, references):
  """Calculates rouge scores for a list of hypotheses and
  references
  
  Args:
    * hypotheses: a list of n sentences
    * references: a list of n sentences
  Returns:
    * rouge-1, rouge-2, rouge-l: list of n tuple (f-measure, precision, recall)
  """

  # Filter out hyps that are of 0 length
  hyps_and_refs = zip(hypotheses, references)
  hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
  hypotheses, references = zip(*hyps_and_refs)

  # Calculate ROUGE-1 F1, precision, recall scores
  rouge_1 = [
      rouge_n([hyp], [ref], 1) for hyp, ref in zip(hypotheses, references)
  ]

  # Calculate ROUGE-2 F1, precision, recall scores
  rouge_2 = [
      rouge_n([hyp], [ref], 2) for hyp, ref in zip(hypotheses, references)
  ]

  # Calculate ROUGE-L F1, precision, recall scores
  rouge_l = [
      rouge_l_sentence_level([hyp], [ref])
      for hyp, ref in zip(hypotheses, references)
  ]

  return rouge_1, rouge_2, rouge_l

def avg_rouge(hypotheses, references):
  """Calculates average rouge scores for a list of hypotheses and
  references
  """
  rouge_1, rouge_2, rouge_l = rouge(hypotheses, references)

  avg_rouge_1 = tuple(map(np.mean, zip(*rouge_1)))
  avg_rouge_2 = tuple(map(np.mean, zip(*rouge_2)))
  avg_rouge_l = tuple(map(np.mean, zip(*rouge_l)))

  return avg_rouge_1, avg_rouge_2, avg_rouge_l
'''
