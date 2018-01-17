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

"""This is the top-level file to train, evaluate or test your summarization model"""

import sys
import time
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data import Vocab
from batcher import Batcher
from model import SentenceSelector
from evaluate import SelectorEval
import util
import pdb

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/eval_all')
tf.app.flags.DEFINE_boolean('single_pass', False, '')

# Where to save output
tf.app.flags.DEFINE_integer('max_train_iter', 29000, 'max iterations to train')
tf.app.flags.DEFINE_integer('save_model_every', 10, 'save the model every N iterations')
tf.app.flags.DEFINE_integer('model_max_to_keep', 3, 'save latest N models')
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# evaluation settings
tf.app.flags.DEFINE_boolean('load_best_val_model', False, '')
tf.app.flags.DEFINE_boolean('load_best_test_model', False, '')
tf.app.flags.DEFINE_string('decode_ckpt_path', '', 'checkpoint path for decoding')
tf.app.flags.DEFINE_float('thres', 0.4, 'threshold of probabilities')
tf.app.flags.DEFINE_integer('min_select_sent', 5, 'min sentences need to be selected')
tf.app.flags.DEFINE_integer('max_select_sent', 20, 'max sentences to be selected')
tf.app.flags.DEFINE_boolean('eval_gt_rouge', False, 'whether to evaluate ROUGE scores')
tf.app.flags.DEFINE_boolean('eval_rouge', False, 'whether to evaluate ROUGE scores')
tf.app.flags.DEFINE_boolean('save_pkl', False, 'whether to save the results as pickle files')
tf.app.flags.DEFINE_boolean('save_bin', True, 'whether to save the results as binary files')
tf.app.flags.DEFINE_boolean('plot', True, 'whether to plot the precision/recall and recall/ratio curves')

# loss
tf.app.flags.DEFINE_string('loss', 'CE', 'CE/FL/PG (cross entropy/focal loss/policy gradient)')
tf.app.flags.DEFINE_float('gamma', 2.0, 'gamma used in focal loss')
tf.app.flags.DEFINE_string('reward', 'r', 'r/f (rougeL recall/f-measure score)')
tf.app.flags.DEFINE_float('regu_ratio', 0.3, 'select ratio used in regularization term of policy gradient')
tf.app.flags.DEFINE_float('regu_ratio_wt', 1.0, 'wieght of ratio regularization term of policy gradient')
tf.app.flags.DEFINE_float('regu_l2_wt', 0.0, 'weight of l2 regularization term of policy gradient')

# Hyperparameters
tf.app.flags.DEFINE_string('rnn_type', 'GRU', 'LSTM/GRU')
tf.app.flags.DEFINE_integer('hidden_dim', 200, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 64, 'minibatch size')
tf.app.flags.DEFINE_integer('max_art_len', 100, 'max timesteps of word-level encoder')
tf.app.flags.DEFINE_integer('max_sent_len', 50, 'max timesteps of sentence-level encoder')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.1, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

def write_to_summary(value, tag_name, step, summary_writer):
  summary = tf.Summary()
  summary.value.add(tag=tag_name, simple_value=value)
  summary_writer.add_summary(summary, step)


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, tag_name, decay=0.99):
  """Calculate the running average loss via exponential decay.
  This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

  Args:
    loss: loss on the most recent eval step
    running_avg_loss: running_avg_loss so far
    summary_writer: FileWriter object to write for tensorboard
    step: training iteration step
    decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

  Returns:
    running_avg_loss: new running average loss
  """
  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  loss_sum = tf.Summary()
  tag_name2 = tag_name + '/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name2, simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  tf.logging.info(tag_name + ': ' + str(running_avg_loss))
  return running_avg_loss

def setup_training(model, batcher):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  default_device = tf.device('/gpu:0')
  with default_device:
    model.build_graph() # build the graph
    params = tf.global_variables()
    vars_to_save = [param for param in params if "Adagrad" not in param.name]
    uninitialized_vars = [param for param in params if "Adagrad" in param.name]
    saver = tf.train.Saver(vars_to_save, max_to_keep=FLAGS.model_max_to_keep) # only keep 1 checkpoint at a time
    local_init_op = tf.variables_initializer(uninitialized_vars)
  
  sv = tf.train.Supervisor(logdir=train_dir,
                     is_chief=True,
                     saver=saver,
                     local_init_op=local_init_op,
                     summary_op=None,
                     save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                     save_model_secs=0, # checkpoint every 60 secs
                     global_step=model.global_step)
  summary_writer = sv.summary_writer
  tf.logging.info("Preparing or waiting for session...")
  sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
  tf.logging.info("Created session.")

  try:
    run_training(model, batcher, sess_context_manager, sv, summary_writer) # this is an infinite loop until interrupted
  except KeyboardInterrupt:
    tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
    sv.stop()


def run_training(model, batcher, sess_context_manager, sv, summary_writer):
  """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
  tf.logging.info("starting run_training")
  ckpt_path = os.path.join(FLAGS.log_root, "train", "model.ckpt")

  with sess_context_manager as sess:
    for _ in range(FLAGS.max_train_iter): # repeats until interrupted
      batch = batcher.next_batch()

      tf.logging.info('running training step...')
      t0=time.time()
      results = model.run_train_step(sess, batch)
      t1=time.time()
      tf.logging.info('seconds for training step: %.3f', t1-t0)

      loss = results['loss']
      tf.logging.info('loss: %f', loss) # print the loss to screen

      if not np.isfinite(loss):
        raise Exception("Loss is not finite. Stopping.")

      train_step = results['global_step'] # we need this to update our running average loss

      if FLAGS.loss == 'PG':
        if FLAGS.regu_ratio_wt > 0.0:
          tf.logging.info('ratio loss: %f', results['ratio_loss'])
          tf.logging.info('total loss: %f', results['total_loss'])
        write_to_summary(results['sample']['avg_reward'], 'SentSelector/reward', train_step, summary_writer)
        write_to_summary(results['sample']['avg_ratio'], 'SentSelector/select_ratio/sample', train_step, summary_writer)
        write_to_summary(results['sample']['avg_gt_recall'], 'SentSelector/gt_recall/sample', train_step, summary_writer)
        tf.logging.info('Sample, reward: %f, ratio: %f, gt_recall: %f' % (results['sample']['avg_reward'], results['sample']['avg_ratio'], results['sample']['avg_gt_recall']))
        #write_to_summary(results['argmax']['avg_reward'], 'SentSelector/reward/argmax', train_step, summary_writer)
        #write_to_summary(results['argmax']['avg_ratio'], 'SentSelector/select_ratio/argmax', train_step, summary_writer)
        #write_to_summary(results['argmax']['avg_gt_recall'], 'SentSelector/gt_recall/argmax', train_step, summary_writer)
        #tf.logging.info('Argmax, reward: %f, ratio: %f, gt_recall: %f' % (results['argmax']['avg_reward'], results['argmax']['avg_ratio'], results['argmax']['avg_gt_recall']))
        write_to_summary(results['running_avg_reward'], 'SentSelector/running_avg_reward/sample/decay=0.990000', train_step, summary_writer)
        tf.logging.info('running_avg_reward: %f', results['running_avg_reward'])

      recall, ratio, _ = util.get_batch_ratio(batch.original_articles_sents, batch.original_extract_sent_ids, results['probs'], target_recall=0.9)
      if recall < 0.89 or recall > 0.91:
        ratio = 1.0
      write_to_summary(ratio, 'SentSelector/select_ratio/recall=0.9', train_step, summary_writer)

      # get the summaries and iteration number so we can write summaries to tensorboard
      summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
      summary_writer.add_summary(summaries, train_step) # write the summaries
      if train_step % 100 == 0: # flush the summary writer every so often
        summary_writer.flush()

      if train_step % FLAGS.save_model_every == 0:
        sv.saver.save(sess, ckpt_path, global_step=train_step)

      print 'Step: ', train_step


def run_eval(model, batcher):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
  model.build_graph() # build the graph
  saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
  sess = tf.Session(config=util.get_config())
  if "val" in FLAGS.data_path: dataset = "val"
  elif "test" in FLAGS.data_path: dataset = "test"
  eval_dir = os.path.join(FLAGS.log_root, "eval_" + dataset) # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  summary_writer = tf.summary.FileWriter(eval_dir)

  if FLAGS.loss == 'PG':
    running_avg_reward_sample = 0 # the eval job keeps a smoother
    best_reward_sample = None  # will hold the best loss achieved so far
    running_avg_reward_argmax = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_reward_argmax = None  # will hold the best loss achieved so far
  else:
    running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_loss = None  # will hold the best loss achieved so far

  train_dir = os.path.join(FLAGS.log_root, "train")
  first_eval_step = True

  while True:
    ckpt_state = tf.train.get_checkpoint_state(train_dir)
    if ckpt_state:
      step = int(os.path.basename(ckpt_state.model_checkpoint_path).split('-')[1])

      if first_eval_step:
        final_step = (int(step/FLAGS.max_train_iter) + 1) * FLAGS.max_train_iter
        first_eval_step = False
      if step == final_step:
        break
    
    #tf.logging.info('max_enc_steps: %d, max_dec_steps: %d', FLAGS.max_enc_steps, FLAGS.max_dec_steps)
    _ = util.load_ckpt(saver, sess) # load a new checkpoint
    batch = batcher.next_batch() # get the next batch

    # run eval on the batch
    t0=time.time()
    results = model.run_eval_step(sess, batch)
    t1=time.time()
    tf.logging.info('seconds for batch: %.2f', t1-t0)

    # print the loss and coverage loss to screen
    loss = results['loss']
    tf.logging.info('loss: %f', loss)
    train_step = results['global_step']

    if FLAGS.loss == 'PG':
      if FLAGS.regu_ratio_wt > 0.0:
        tf.logging.info('ratio loss: %f', results['ratio_loss'])
        tf.logging.info('total loss: %f', results['total_loss'])
      write_to_summary(results['sample']['avg_reward'], 'SentSelector/reward', train_step, summary_writer)
      write_to_summary(results['sample']['avg_ratio'], 'SentSelector/select_ratio/sample', train_step, summary_writer)
      write_to_summary(results['sample']['avg_gt_recall'], 'SentSelector/gt_recall/sample', train_step, summary_writer)
      tf.logging.info('Sample, reward: %f, ratio: %f, gt_recall: %f' % (results['sample']['avg_reward'], results['sample']['avg_ratio'], results['sample']['avg_gt_recall']))
      write_to_summary(results['argmax']['avg_reward'], 'SentSelector/reward/argmax', train_step, summary_writer)
      write_to_summary(results['argmax']['avg_ratio'], 'SentSelector/select_ratio/argmax', train_step, summary_writer)
      write_to_summary(results['argmax']['avg_gt_recall'], 'SentSelector/gt_recall/argmax', train_step, summary_writer)
      tf.logging.info('Argmax, reward: %f, ratio: %f, gt_recall: %f' % (results['argmax']['avg_reward'], results['argmax']['avg_ratio'], results['argmax']['avg_gt_recall']))

    recall, ratio, _ = util.get_batch_ratio(batch.original_articles_sents, batch.original_extract_sent_ids, results['probs'], target_recall=0.9)
    if recall < 0.89 or recall > 0.91:
      ratio = 1.0
    write_to_summary(ratio, 'SentSelector/select_ratio/recall=0.9', train_step, summary_writer)

    # add summaries
    summaries = results['summaries']
    summary_writer.add_summary(summaries, train_step)

    # calculate running avg loss
    if FLAGS.loss == 'PG':
      running_avg_reward_sample = calc_running_avg_loss(results['sample']['avg_reward'], running_avg_reward_sample, summary_writer, train_step, 'SentSelector/running_avg_reward/sample')
      running_avg_reward_argmax = calc_running_avg_loss(results['argmax']['avg_reward'], running_avg_reward_argmax, summary_writer, train_step, 'SentSelector/running_avg_reward/argmax')
    else:
      #running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)
      running_avg_loss = calc_running_avg_loss(ratio, running_avg_loss, summary_writer, train_step, 'running_avg_ratio')

    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    if FLAGS.loss == 'PG':
      if best_reward_argmax is None or running_avg_reward_argmax > best_reward_argmax:
        tf.logging.info('Found new best model with %.3f running_avg_reward. Saving to %s', running_avg_reward_argmax, bestmodel_save_path)
        saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
        best_reward_sample = running_avg_reward_sample
    else:
      if best_loss is None or running_avg_loss < best_loss:
        tf.logging.info('Found new best model with %.3f running_avg_ratio. Saving to %s', running_avg_loss, bestmodel_save_path)
        saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
        best_loss = running_avg_loss

    # flush the summary writer every so often
    if train_step % 100 == 0:
      summary_writer.flush()


def main(unused_argv):
  if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
    raise Exception("Problem with flags: %s" % unused_argv)

  tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
  tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
  if not os.path.exists(FLAGS.log_root):
    if FLAGS.mode=="train":
      os.makedirs(FLAGS.log_root)
    else:
      raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

  vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size) # create a vocabulary

  # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  hparam_list = ['mode', 'rnn_type', 'loss', 'gamma', 'reward', 'regu_ratio', 'regu_ratio_wt', 'regu_l2_wt', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_art_len', 'max_sent_len']
  hps_dict = {}
  for key,val in FLAGS.__flags.iteritems(): # for each flag
    if key in hparam_list: # if it's in the list
      hps_dict[key] = val # add it to the dict
  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

  # Create a batcher object that will create minibatches of data
  if hps.mode == 'eval_all':
    hps = hps._replace(batch_size=1)
  batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)
  tf.set_random_seed(111) # a seed value for randomness

  if hps.mode == 'train':
    print "creating model..."
    model = SentenceSelector(hps, vocab)
    setup_training(model, batcher)
  elif hps.mode == 'eval':
    model = SentenceSelector(hps, vocab)
    run_eval(model, batcher)
  elif hps.mode == 'eval_all':
    model = SentenceSelector(hps, vocab)
    evaluator = SelectorEval(model, batcher, vocab)
    evaluator.evaluate()
  else:
    raise ValueError("The 'mode' flag must be one of train/eval/decode")

if __name__ == '__main__':
  tf.app.run()
