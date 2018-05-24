import os
import time
import tensorflow as tf
from tensorflow.core.example import example_pb2
import struct
import data
import cPickle as pk
import pyrouge
import util
import logging
import numpy as np

FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
CHUNK_SIZE = 1000

class SelectorEvaluator(object):

  def __init__(self, model, batcher, vocab):
    """Initialize decoder.

    Args:
      model: a SentSelector object.
      batcher: a Batcher object.
      vocab: Vocabulary object
    """
    # get the data split set
    if "train" in FLAGS.data_path: self._dataset = "train"
    elif "val" in FLAGS.data_path: self._dataset = "val"
    elif "test" in FLAGS.data_path: self._dataset = "test"
    else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))

    # create the data loader
    self._batcher = batcher

    if FLAGS.eval_gt_rouge: # no need to load model
      # Make a descriptive decode directory name
      self._decode_dir = os.path.join(FLAGS.log_root, 'select_gt' + self._dataset)
      tf.logging.info('Save evaluation results to '+ self._decode_dir)
      if os.path.exists(self._decode_dir):
        raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

      # Make the decode dir
      os.makedirs(self._decode_dir)

      # Make the dirs to contain output written in the correct format for pyrouge
      self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
      if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
      self._rouge_gt_dir = os.path.join(self._decode_dir, "gt_selected")
      if not os.path.exists(self._rouge_gt_dir): os.mkdir(self._rouge_gt_dir)
    else:
      self._model = model
      self._model.build_graph()
      self._vocab = vocab
      self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
      self._sess = tf.Session(config=util.get_config())

      # Load an initial checkpoint to use for decoding
      if FLAGS.load_best_eval_model:
        tf.logging.info('Loading best eval checkpoint')
        ckpt_path = util.load_ckpt(self._saver, self._sess, ckpt_dir='eval')
      elif FLAGS.eval_ckpt_path:
        ckpt_path = util.load_ckpt(self._saver, self._sess, ckpt_path=FLAGS.eval_ckpt_path)
      else:
        tf.logging.info('Loading best train checkpoint')
        ckpt_path = util.load_ckpt(self._saver, self._sess)

      if FLAGS.single_pass:
        # Make a descriptive decode directory name
        ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
        decode_root_dir, decode_dir = get_decode_dir_name(ckpt_name, self._dataset)
        self._decode_root_dir = os.path.join(FLAGS.log_root, decode_root_dir)
        self._decode_dir = os.path.join(FLAGS.log_root, decode_root_dir, decode_dir)
        tf.logging.info('Save evaluation results to '+ self._decode_dir)
        if os.path.exists(self._decode_dir):
          raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)
      else: # Generic decode dir name
        self._decode_dir = os.path.join(FLAGS.log_root, "select")

      # Make the decode dir if necessary
      if not os.path.exists(self._decode_dir): os.makedirs(self._decode_dir)

      if FLAGS.single_pass:
        # Make the dirs to contain output written in the correct format for pyrouge
        self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
        if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
        self._rouge_dec_dir = os.path.join(self._decode_dir, "selected")
        if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)
        if FLAGS.save_pkl:
          self._result_dir = os.path.join(self._decode_dir, "select_result")
          if not os.path.exists(self._result_dir): os.mkdir(self._result_dir)

        self._probs_pkl_path = os.path.join(self._decode_root_dir, "probs.pkl")
        if not os.path.exists(self._probs_pkl_path): 
          self._make_probs_pkl = True
        else:
          self._make_probs_pkl = False
        self._precision = []
        self._recall = []
        self._accuracy = []
        self._ratio = []
        self._select_sent_num = []


  def evaluate(self):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0

    if not FLAGS.eval_gt_rouge:
      if not self._make_probs_pkl:
        with open(self._probs_pkl_path, 'rb') as probs_file:
          probs_all = pk.load(probs_file)
      else:
        probs_all = {'probs': {}, 'article': {}, 'reference': {}, 'gt_ids': {}}

    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      if batch is None: # finished decoding dataset in single_pass mode
        assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")

        if FLAGS.eval_gt_rouge:
          tf.logging.info("Output has been saved in %s and %s. Starting ROUGE eval...", self._rouge_ref_dir, self._rouge_gt_dir)
          rouge_results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_gt_dir)
          rouge_log(rouge_results_dict, self._decode_dir, 'ROUGE_results_gt_select.txt')
        else:
          results_log(self._precision, self._recall, self._accuracy, self._select_sent_num, self._ratio, self._decode_dir)
          if self._make_probs_pkl:
            with open(self._probs_pkl_path, 'wb') as output_file:
              pk.dump(probs_all, output_file)

          tf.logging.info("Output has been saved in %s and %s. Starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
          rouge_results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
          rouge_log(rouge_results_dict, self._decode_dir, 'ROUGE_results.txt')

        t1 = time.time()
        tf.logging.info("evaluation time: %.3f min", (t1-t0)/60.0)
        return

      original_article_sents = batch.original_articles_sents[0]  # string
      original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings
      original_select_ids = batch.original_extracts_ids[0]
      original_select_sents = [original_article_sents[idx] for idx in original_select_ids]

      if FLAGS.eval_gt_rouge:
        self.write_for_rouge(original_abstract_sents, original_select_sents, counter) # write ref summary and decoded summary to file, to eval with pyrouge later
        counter += 1
        continue

      # Run inference or read saved results to get model probabilities
      if self._make_probs_pkl:
        output = self._model.run_eval_step(self._sess, batch, probs_only=True)
        probs = output['probs'][0]
        probs_all['probs'][counter] = probs
        probs_all['article'][counter] = original_article_sents
        probs_all['reference'][counter] = original_abstract_sents
        probs_all['gt_ids'][counter] = original_select_ids
      else:
        probs = probs_all['probs'][counter]

      # calculate precision and recall
      min_select = FLAGS.min_select_sent if FLAGS.select_method != 'num' else None
      max_select = FLAGS.max_select_sent if FLAGS.select_method != 'num' else None
      select_sents, select_ids, p, r, acc, ratio = util.get_select_accuracy_one_thres(original_article_sents, \
                                                             probs, original_select_ids, FLAGS.thres, \
                                                             min_select=min_select, \
                                                             max_select=max_select,\
                                                             method=FLAGS.select_method)
      if FLAGS.select_method in ['prob', 'ratio']:
        assert len(select_ids) <= FLAGS.max_select_sent
        assert len(select_ids) >= FLAGS.min_select_sent or len(select_ids) == len(original_article_sents)
      elif FLAGS.select_method == 'num':
        assert len(select_ids) <= int(FLAGS.thres)
      self._precision.append(p)
      self._recall.append(r)
      self._accuracy.append(acc)
      self._select_sent_num.append(len(select_ids))
      self._ratio.append(ratio)
      
      if FLAGS.single_pass:
        self.write_for_rouge(original_abstract_sents, select_sents, counter) # write ref summary and decoded summary to file, to eval with pyrouge later
        if FLAGS.save_pkl:
          self.save_result(original_article_sents, original_abstract_sents, select_sents, \
                           select_ids, original_select_ids, p, r, acc, counter)
        counter += 1 # this is how many examples we've decoded
 

  def save_result(self, article_sents, reference_sents, select_sents, select_ids, gt_ids, precision, recall, acc, index):
    """save the result in pickle format"""
    data = {'article': article_sents, 
            'reference': reference_sents, 
            'selected': select_sents, 
            'gt_ids': gt_ids, 
            'selected_ids': select_ids,
            'precision': precision,
            'recall': recall,
            'accuracy': acc}
    output_fname = os.path.join(self._result_dir, 'result_%06d.pkl' % index)
    with open(output_fname, 'wb') as output_file:
      pk.dump(data, output_file)
    tf.logging.info('Wrote result data to %s', output_fname)

  def write_for_rouge(self, reference_sents, decoded_sents, ex_index):
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
    if FLAGS.eval_gt_rouge:
      decoded_file = os.path.join(self._rouge_gt_dir, "%06d_decoded.txt" % ex_index)
    else:
      decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
      for idx,sent in enumerate(reference_sents):
        f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")
    with open(decoded_file, "w") as f:
      for idx,sent in enumerate(decoded_sents):
        f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")

    tf.logging.info("Wrote example %i to file" % ex_index)


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

def rouge_log(results_dict, dir_to_write, output_file):
  """Log ROUGE results to screen and write to file.

  Args:
    results_dict: the dictionary returned by pyrouge
    dir_to_write: the directory where we will write the results to"""
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
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  tf.logging.info(log_str) # log to screen
  results_file = os.path.join(dir_to_write, output_file)
  tf.logging.info("Writing final ROUGE results to %s...", results_file)
  with open(results_file, "w") as f:
    f.write(log_str)

def results_log(precision_list, recall_list, accuracy_list, sent_num_list, ratio_list, dir_to_write):
  avg_p = sum(precision_list)/len(precision_list)
  avg_r = sum(recall_list)/len(recall_list)
  avg_a = sum(accuracy_list)/len(accuracy_list)
  avg_sent_num = float(sum(sent_num_list))/len(sent_num_list)
  avg_ratio = sum(ratio_list)/len(ratio_list)
  tf.logging.info("Average precision: %.3f", avg_p)
  tf.logging.info("Average recall: %.3f", avg_r)
  tf.logging.info("Average accuracy: %.3f", avg_a)
  tf.logging.info("Average number of sentences: %.3f", avg_sent_num)
  tf.logging.info("Average ratio: %.3f", avg_ratio)
  results_file = os.path.join(dir_to_write, "results.txt")
  log_str = ""
  log_str += "Average precision: %.3f\n" % avg_p
  log_str += "Average recall: %.3f\n" % avg_r
  log_str += "Average accuracy: %.3f\n" % avg_a
  log_str += "Average number of sentences: %.3f\n" % avg_sent_num
  log_str += "Average ratio: %.3f\n" % avg_ratio
  with open(results_file, "w") as f:
    f.write(log_str)


def get_decode_dir_name(ckpt_name, dataset):
  """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""
  root_name = "select_%s_%imaxart_%imaxsent" % (dataset, FLAGS.max_art_len, FLAGS.max_sent_len)
  if ckpt_name is not None:
    root_name += "_%s" % ckpt_name
  if FLAGS.select_method == 'num':
    dirname = "%s_%dthres" % (FLAGS.select_method, FLAGS.thres)
  else:
    dirname = "%s_%.2fthres_%iminselect_%imaxselect" % (FLAGS.select_method, FLAGS.thres, FLAGS.min_select_sent, FLAGS.max_select_sent)
  return root_name, dirname

