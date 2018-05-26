# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to run beam search decoding, including running ROUGE evaluation and producing JSON datafiles for the in-browser attention visualizer, which can be found here https://github.com/abisee/attn_vis"""

import os
import time
import tensorflow as tf
from batcher import Batcher
import beam_search
import data
import cPickle as pk
import json
import pyrouge
import util
import logging
import numpy as np
import pdb

FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint

class BeamSearchDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, batcher, vocab):
    """Initialize decoder.

    Args:
      model: a Seq2SeqAttentionModel object.
      batcher: a Batcher object.
      vocab: Vocabulary object
    """
    self._model = model
    self._model.build_graph()
    self._batcher = batcher
    self._vocab = vocab
    self._saver = tf.train.Saver(max_to_keep=3) # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())
    if FLAGS.mode == 'evalall':
      self.prepare_evaluate()

  def prepare_evaluate(self, ckpt_path=None):
    # Load an initial checkpoint to use for decoding
    if FLAGS.mode == 'evalall':
      if FLAGS.load_best_eval_model:
        tf.logging.info('Loading best eval checkpoint')
        ckpt_path = util.load_ckpt(self._saver, self._sess, ckpt_dir='eval_'+FLAGS.eval_method)
      elif FLAGS.eval_ckpt_path:
        ckpt_path = util.load_ckpt(self._saver, self._sess, ckpt_path=FLAGS.eval_ckpt_path)
      else:
        tf.logging.info('Loading best train checkpoint')
        ckpt_path = util.load_ckpt(self._saver, self._sess)
    elif FLAGS.mode == 'eval':
      _ = util.load_ckpt(self._saver, self._sess, ckpt_path=ckpt_path) # load a new checkpoint

    if FLAGS.single_pass:
      # Make a descriptive decode directory name
      ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
      self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
      tf.logging.info('Save evaluation results to '+ self._decode_dir)
      if os.path.exists(self._decode_dir):
        if FLAGS.mode == 'eval':
          return False  # The checkpoint has already been evaluated. Evaluate next one.
        else:
          raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)
    else: # Generic decode dir name
      self._decode_dir = os.path.join(FLAGS.log_root, "decode")

    # Make the decode dir if necessary
    if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

    if FLAGS.single_pass:
      # Make the dirs to contain output written in the correct format for pyrouge
      self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
      if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
      self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
      if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)
      if FLAGS.save_vis:
        self._rouge_vis_dir = os.path.join(self._decode_dir, "visualize")
        if not os.path.exists(self._rouge_vis_dir): os.mkdir(self._rouge_vis_dir)
      if FLAGS.save_pkl:
        self._result_dir = os.path.join(self._decode_dir, "result")
        if not os.path.exists(self._result_dir): os.mkdir(self._result_dir)
    return True

  def evaluate(self):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0

    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      if batch is None: # finished decoding dataset in single_pass mode
        assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        tf.logging.info("Output has been saved in %s and %s. Starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_results, rouge_results_str = rouge_log(rouge_results_dict, self._decode_dir)
        t1 = time.time()
        tf.logging.info("evaluation time: %.3f min", (t1-t0)/60.0)
        return rouge_results, rouge_results_str

      if FLAGS.decode_method == 'greedy':
        output_ids = self._model.run_greedy_search(self._sess, batch)
        for i in range(FLAGS.batch_size):
          self.process_one_article(batch.original_articles_sents[i], batch.original_abstracts_sents[i], \
                                   batch.original_extracts_ids[i], output_ids[i], \
                                   batch.art_oovs[i], None, None, None, counter)
          counter += 1
      elif FLAGS.decode_method == 'beam':
        # Run beam search to get best Hypothesis
        best_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)

        # Extract the output ids from the hypothesis and convert back to words
        output_ids = [int(t) for t in best_hyp.tokens[1:]]    # remove start token
        best_hyp.log_probs = best_hyp.log_probs[1:]   # remove start token probability
        self.process_one_article(batch.original_articles_sents[0], batch.original_abstracts_sents[0], \
                                 batch.original_extracts_ids[0], output_ids, batch.art_oovs[0], \
                                 best_hyp.attn_dists, best_hyp.p_gens, best_hyp.log_probs, counter)
        counter += 1

  def process_one_article(self, original_article_sents, original_abstract_sents, \
                          original_selected_ids, output_ids, oovs, \
                          attn_dists, p_gens, log_probs, counter):
    # Remove the [STOP] token from decoded_words, if necessary
    decoded_words = data.outputids2words(output_ids, self._vocab, oovs)
    try:
      fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
      decoded_words = decoded_words[:fst_stop_idx]
    except ValueError:
      decoded_words = decoded_words
    decoded_output = ' '.join(decoded_words) # single string
    decoded_sents = data.words2sents(decoded_words)

    if FLAGS.single_pass:
      verbose = False if FLAGS.mode == 'eval' else True
      self.write_for_rouge(original_abstract_sents, decoded_sents, counter, verbose) # write ref summary and decoded summary to file, to eval with pyrouge later
      if FLAGS.decode_method == 'beam' and FLAGS.save_vis:
        original_article = ' '.join(original_article_sents)
        original_abstract = ' '.join(original_abstract_sents)
        article_withunks = data.show_art_oovs(original_article, self._vocab) # string
        abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, oovs)
        self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, \
                               attn_dists, p_gens, log_probs, counter, verbose)
      if FLAGS.save_pkl:
        self.save_result(original_article_sents, original_abstract_sents, \
                         original_selected_ids, decoded_sents, counter, verbose)

  def save_result(self, article_sents, reference_sents, select_ids, decoded_sents, index, verbose=False):
    """save the result in pickle format"""
    data = {'article': article_sents, 
            'reference': reference_sents, 
            'select_ids': select_ids, 
            'decoded': decoded_sents}
    output_fname = os.path.join(self._result_dir, 'result_%06d.pkl' % index)
    with open(output_fname, 'wb') as output_file:
      pk.dump(data, output_file)
    if verbose:
      tf.logging.info('Wrote result data to %s', output_fname)

  def write_for_rouge(self, reference_sents, decoded_sents, ex_index, verbose=False):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
    """
    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    # Write to file
    ref_file = os.path.join(self._rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
      for idx,sent in enumerate(reference_sents):
        f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")
    with open(decoded_file, "w") as f:
      for idx,sent in enumerate(decoded_sents):
        f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")

    if verbose:
      tf.logging.info("Wrote example %i to file" % ex_index)


  def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens, \
                        log_probs, count=None, verbose=False):
    """Write some data to json file, which can be read into the in-browser attention visualizer tool:
      https://github.com/abisee/attn_vis

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      attn_dists: List of arrays; the attention distributions.
      decoded_words: List of strings; the words of the generated summary.
      p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
    """
    article_lst = article.split() # list of words
    decoded_lst = decoded_words # list of decoded words
    to_write = {
        'article_lst': [make_html_safe(t) for t in article_lst],
        'decoded_lst': [make_html_safe(t) for t in decoded_lst],
        'abstract_str': make_html_safe(abstract),
        'attn_dists': attn_dists,
        'probs': np.exp(log_probs).tolist()
    }
    to_write['p_gens'] = p_gens
    if count != None:
      output_fname = os.path.join(self._rouge_vis_dir, 'attn_vis_data_%06d.json' % count)
    else:
      output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
    with open(output_fname, 'w') as output_file:
      json.dump(to_write, output_file)
    if verbose:
      tf.logging.info('Wrote visualization data to %s', output_fname)

  def init_batcher(self):
    self._batcher = Batcher(FLAGS.data_path, self._vocab, self._model._hps, single_pass=FLAGS.single_pass)


def print_results(article, abstract, decoded_output):
  """Prints the article, the reference summmary and the decoded summary to screen"""
  print ""
  tf.logging.info('ARTICLE:  %s', article)
  tf.logging.info('REFERENCE SUMMARY: %s', abstract)
  tf.logging.info('GENERATED SUMMARY: %s', decoded_output)
  print ""


def make_html_safe(s):
  """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s


def rouge_eval(ref_dir, dec_dir):
  """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
  """Log ROUGE results to screen and write to file.

  Args:
    results_dict: the dictionary returned by pyrouge
    dir_to_write: the directory where we will write the results to"""
  rouge_results = {}
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      if y == 'f_score':
        rouge_results[x] = val
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  tf.logging.info(log_str) # log to screen
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  tf.logging.info("Writing final ROUGE results to %s...", results_file)
  with open(results_file, "w") as f:
    f.write(log_str)
  return rouge_results, log_str

def get_decode_dir_name(ckpt_name):
  """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

  if "train" in FLAGS.data_path: dataset = "train"
  elif "val" in FLAGS.data_path: dataset = "val"
  elif "test" in FLAGS.data_path: dataset = "test"
  else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
  dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
  if ckpt_name is not None:
    dirname += "_%s_%s" % (ckpt_name, FLAGS.decode_method)
  return dirname
