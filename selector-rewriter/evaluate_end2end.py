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
import pdb

FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
CHUNK_SIZE = 1000

class SelectorEvaluator(object):

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
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt(self._saver, self._sess, FLAGS.eval_ckpt_path)

    if FLAGS.single_pass:
      # Make a descriptive decode directory name
      ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
      decode_root_dir, decode_dir, self._dataset = get_decode_dir_name(ckpt_name)
      self._decode_root_dir = os.path.join(FLAGS.log_root, decode_root_dir)
      self._decode_dir = os.path.join(FLAGS.log_root, decode_root_dir, decode_dir)
      if os.path.exists(self._decode_dir):
        raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)
    else: # Generic decode dir name
      self._decode_dir = os.path.join(FLAGS.log_root, "select")

    # Make the decode dir if necessary
    if not os.path.exists(self._decode_dir): os.makedirs(self._decode_dir)

    if FLAGS.single_pass:
      # Make the dirs to contain output written in the correct format for pyrouge
      if self._dataset != 'train':
        self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
        if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
        self._rouge_dec_dir = os.path.join(self._decode_dir, "selected")
        if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)
        self._result_dir = os.path.join(self._decode_dir, "select_result")
        if not os.path.exists(self._result_dir): os.mkdir(self._result_dir)
      self._chunks_dir = os.path.join(self._decode_dir, "chunked")
      if not os.path.exists(self._chunks_dir): os.mkdir(self._chunks_dir)

      self._probs_pkl_path = os.path.join(self._decode_root_dir, "probs.pkl")
      if not os.path.exists(self._probs_pkl_path): 
        self._make_probs_pkl = True
      else:
        self._make_probs_pkl = False
      self._precision = []
      self._recall = []
      self._select_sent_num = []


  def evaluate(self):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0

    '''
    rouge_dir = './log/cnn_dailymail/decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-111000/'
    print('Start rouge eval. Dir: ', rouge_dir)
    results_dict = rouge_eval(rouge_dir + 'reference', rouge_dir + 'decoded')
    rouge_log(results_dict, rouge_dir)
    return
    '''
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
        results_log(self._precision, self._recall, self._select_sent_num, self._decode_dir)
        chunk_file(self._decode_dir, self._chunks_dir, self._dataset)

        if self._make_probs_pkl:
          with open(self._probs_pkl_path, 'wb') as output_file:
            pk.dump(probs_all, output_file)

        if self._dataset != 'train':
          tf.logging.info("Output has been saved in %s and %s. Starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
          rouge_results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
          rouge_log(rouge_results_dict, self._decode_dir)
        t1 = time.time()
        tf.logging.info("evaluation time: %.3f min", (t1-t0)/60.0)
        return

      original_article_sents = batch.original_articles_sents[0]  # string
      original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings
      original_selected_ids = batch.original_extract_sent_ids[0]

      # Run beam search to get best Hypothesis
      if self._make_probs_pkl:
        output = self._model.run_eval_step(self._sess, batch, probs_only=True)
        probs = output['probs'][0]
        probs_all['probs'][counter] = probs
        probs_all['article'][counter] = original_article_sents
        probs_all['reference'][counter] = original_abstract_sents
        probs_all['gt_ids'][counter] = original_selected_ids
      else:
        probs = probs_all['probs'][counter]
      selected_sents, selected_ids, p, r = util.get_select_accuracy_one_thres(original_article_sents, \
                                                             probs, original_selected_ids, FLAGS.thres, \
                                                             min_select=FLAGS.min_select_sent, \
                                                             max_select=FLAGS.max_select_sent)
      assert len(selected_ids) <= FLAGS.max_select_sent
      assert len(selected_ids) >= FLAGS.min_select_sent or len(selected_ids) == len(original_article_sents)
      self._precision.append(p)
      self._recall.append(r)
      self._select_sent_num.append(len(selected_ids))
      
      if FLAGS.single_pass:
        if self._dataset != 'train':
          self.write_for_rouge(original_abstract_sents, selected_sents, counter) # write ref summary and decoded summary to file, to eval with pyrouge later
          self.save_result(original_article_sents, original_abstract_sents, selected_sents, \
                           selected_ids, original_selected_ids, p, r, counter)
        self.write_to_bin(original_article_sents, original_abstract_sents, selected_sents, \
                          selected_ids, original_selected_ids, counter)
        counter += 1 # this is how many examples we've decoded
 

  def save_result(self, article_sents, reference_sents, selected_sents, selected_ids, gt_ids, precision, recall, index):
    """save the result in pickle format"""
    data = {'article': article_sents, 
            'reference': reference_sents, 
            'selected': selected_sents, 
            'gt_ids': gt_ids, 
            'selected_ids': selected_ids,
            'precision': precision,
            'recall': recall}
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
    decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
      for idx,sent in enumerate(reference_sents):
        f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")
    with open(decoded_file, "w") as f:
      for idx,sent in enumerate(decoded_sents):
        f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")

    tf.logging.info("Wrote example %i to file" % ex_index)


  def write_to_bin(self, article_sents, abstract_sents, select_sents, select_ids, gt_ids, ex_index):
    SENTENCE_START = '<s>'
    SENTENCE_END = '</s>'
    article = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in article_sents])
    abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in abstract_sents])
    extract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in select_sents])
    extract_ids = ','.join([str(i) for i in select_ids])
    gt_ids = ','.join([str(i) for i in gt_ids])

    bin_file = os.path.join(self._decode_dir, self._dataset + ".bin")
    with open(bin_file, 'ab+') as writer:
      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([article])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
      tf_example.features.feature['extract'].bytes_list.value.extend([extract])
      tf_example.features.feature['extract_ids'].bytes_list.value.extend([extract_ids])
      tf_example.features.feature['extract_gt_ids'].bytes_list.value.extend([gt_ids])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))
    tf.logging.info("Wrote example %i to binary file" % ex_index)


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
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  tf.logging.info("Writing final ROUGE results to %s...", results_file)
  with open(results_file, "w") as f:
    f.write(log_str)

def results_log(precision_list, recall_list, sent_num_list, dir_to_write):
  avg_p = sum(precision_list)/len(precision_list)
  avg_r = sum(recall_list)/len(recall_list)
  avg_sent_num = float(sum(sent_num_list))/len(sent_num_list)
  tf.logging.info("Average precision: %.3f", avg_p)
  tf.logging.info("Average recall: %.3f", avg_r)
  tf.logging.info("Average number of sentences: %.3f", avg_sent_num)
  results_file = os.path.join(dir_to_write, "results.txt")
  log_str = ""
  log_str += "Average precision: %.3f\n" % avg_p
  log_str += "Average recall: %.3f\n" % avg_r
  log_str += "Average number of sentences: %.3f\n" % avg_sent_num
  with open(results_file, "w") as f:
    f.write(log_str)


def get_decode_dir_name(ckpt_name):
  """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

  if "train" in FLAGS.data_path: dataset = "train"
  elif "val" in FLAGS.data_path: dataset = "val"
  elif "test" in FLAGS.data_path: dataset = "test"
  else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
  root_name = "select_%s_%imaxart_%imaxsent" % (dataset, FLAGS.max_art_len, FLAGS.max_sent_len)
  if ckpt_name is not None:
    root_name += "_%s" % ckpt_name
  dirname = "%.2fthres_%iminselect_%imaxselect" % (FLAGS.thres, FLAGS.min_select_sent, FLAGS.max_select_sent)
  return root_name, dirname, dataset


def chunk_file(input_dir, chunks_dir, set_name):
  in_file = input_dir + '/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1
