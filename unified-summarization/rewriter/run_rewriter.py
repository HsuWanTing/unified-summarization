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
import cPickle as pk
import util
import pdb

FLAGS = tf.app.flags.FLAGS

def write_to_summary(value, tag_name, step, summary_writer):
  summary = tf.Summary()
  summary.value.add(tag=tag_name, simple_value=value)
  summary_writer.add_summary(summary, step)


def convert_to_coverage_model():
  """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
  tf.logging.info("converting non-coverage model to coverage model..")

  # initialize an entire coverage model from scratch
  sess = tf.Session(config=util.get_config())
  print "initializing everything..."
  sess.run(tf.global_variables_initializer())

  # load all non-coverage weights from checkpoint
  saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
  print "restoring non-coverage variables..."
  curr_ckpt = util.load_ckpt(saver, sess)
  print "restored."

  # save this model and quit
  ckpt_path = os.path.join(FLAGS.log_root, "train", "model.ckpt_cov")
  step = curr_ckpt.split('-')[1]
  new_fname = ckpt_path + '-' + step + '-init'
  print "saving model to %s..." % (new_fname)
  new_saver = tf.train.Saver() # this one will save all variables that now exist
  new_saver.save(sess, new_fname)
  print "saved."
  exit()


def setup_training(model, batcher):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  default_device = tf.device('/gpu:0')
  with default_device:
    model.build_graph() # build the graph
    if FLAGS.convert_to_coverage_model:
      assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
      convert_to_coverage_model()

    saver = tf.train.Saver(max_to_keep=FLAGS.model_max_to_keep) # only keep 1 checkpoint at a time

  sv = tf.train.Supervisor(logdir=train_dir,
                     is_chief=True,
                     saver=saver,
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
  if FLAGS.coverage:
    ckpt_path = os.path.join(FLAGS.log_root, "train", "model.ckpt_cov")
  else:
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

      tf.logging.info("pgen_avg: %f", results['p_gen_avg'])

      if FLAGS.coverage:
        tf.logging.info("coverage_loss: %f", results['coverage_loss']) # print the coverage loss to screen

      # get the summaries and iteration number so we can write summaries to tensorboard
      summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
      train_step = results['global_step'] # we need this to update our running average loss
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
  eval_dir = os.path.join(FLAGS.log_root, "eval_loss") # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  summary_writer = tf.summary.FileWriter(eval_dir)
  running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
  best_loss = None  # will hold the best loss achieved so far
  train_dir = os.path.join(FLAGS.log_root, "train")

  while True:
    ckpt_state = tf.train.get_checkpoint_state(train_dir)
    tf.logging.info('max_enc_steps: %d, max_dec_steps: %d', FLAGS.max_enc_steps, FLAGS.max_dec_steps)
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

    tf.logging.info("pgen_avg: %f", results['p_gen_avg'])

    if FLAGS.coverage:
      tf.logging.info("coverage_loss: %f", results['coverage_loss'])

    # add summaries
    summaries = results['summaries']
    summary_writer.add_summary(summaries, train_step)

    # calculate running avg loss
    running_avg_loss = util.calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step, 'running_avg_loss')

    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    if best_loss is None or running_avg_loss < best_loss:
      tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
      saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
      best_loss = running_avg_loss

    # flush the summary writer every so often
    if train_step % 100 == 0:
      summary_writer.flush()

def run_eval_rouge(evaluator):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
  eval_dir = os.path.join(FLAGS.log_root, 'eval_rouge') # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  summary_writer = tf.summary.FileWriter(eval_dir)

  best_rouge_file = os.path.join(eval_dir, 'best_rouge.pkl')
  if os.path.exists(best_rouge_file):
    best_rouges = pk.load(open(best_rouge_file, 'rb'))
    current_step = best_rouges['step']
    best_rouge1 = best_rouges['1']
    best_rouge2 = best_rouges['2']
    best_rougeL = best_rouges['l']
    tf.logging.info('previous best rouge1: %3f, rouge2: %3f, rougeL: %3f, step: %d', \
                    best_rouge1, best_rouge2, best_rougeL, current_step)
  else:
    current_step = None
    best_rouge1 = None  # will hold the best rouge1 achieved so far
    best_rouge2 = None  # will hold the best rouge2 achieved so far
    best_rougeL = None  # will hold the best rougeL achieved so far

  train_dir = os.path.join(FLAGS.log_root, "train")
  if FLAGS.coverage:
    ckpt_base_path = os.path.join(train_dir, "model.ckpt_cov")
  else:
    ckpt_base_path = os.path.join(train_dir, "model.ckpt")

  while True:
    if current_step is None:
      ckpt_state = tf.train.get_checkpoint_state(train_dir)
      if ckpt_state:
        step = os.path.basename(ckpt_state.model_checkpoint_path).split('-')[1]

        if int(step) < FLAGS.start_eval_rouge:
          tf.logging.info('Step = ' + str(step) + ' (smaller than start_eval_rouge, Sleeping for 10 secs...)')
          time.sleep(10)
          continue
        else:
          current_step = int(step)
          current_ckpt_path = ckpt_base_path + '-' + str(current_step)
      else:
        tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", train_dir, 10)
        time.sleep(10)
        continue
    else:
      current_step += FLAGS.save_model_every
      if int(current_step) < FLAGS.start_eval_rouge:
        tf.logging.info('Step = ' + str(current_step) + ' (smaller than start_eval_rouge, Sleeping for 10 secs...)')
        time.sleep(10)
        continue
      current_ckpt_path = ckpt_base_path + '-' + str(current_step)
      evaluator.init_batcher()

    tf.logging.info('max_enc_steps: %d, max_dec_steps: %d', FLAGS.max_enc_steps, FLAGS.max_dec_steps)
    do_eval = evaluator.prepare_evaluate(ckpt_path=current_ckpt_path)
    if not do_eval:  # The checkpoint has already been evaluated. Evaluate next one.
      tf.logging.info('step %d checkpoint has already been evaluated, evaluate next checkpoint.', current_step)
      continue
    rouge_results, rouge_results_str = evaluator.evaluate()

    # print the loss and coverage loss to screen
    results_file = os.path.join(eval_dir, "ROUGE_results_all.txt")
    with open(results_file, "a") as f:
      f.write('Step: ' + str(current_step))
      f.write(rouge_results_str + '\n')

    # add summaries
    write_to_summary(rouge_results['1'], 'rouge_results/rouge1', current_step, summary_writer)
    write_to_summary(rouge_results['2'], 'rouge_results/rouge2', current_step, summary_writer)
    write_to_summary(rouge_results['l'], 'rouge_results/rougeL', current_step, summary_writer)

    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    better_metric = 0
    if best_rouge1 is None or rouge_results['1'] >= best_rouge1:
      best_rouge1 = rouge_results['1']
      better_metric += 1
    if best_rouge2 is None or rouge_results['2'] >= best_rouge2:
      best_rouge2 = rouge_results['2']
      better_metric += 1
    if best_rougeL is None or rouge_results['l'] >= best_rougeL:
      best_rougeL = rouge_results['l']
      better_metric += 1

    if better_metric >= 2:
      tf.logging.info('Found new best model with rouge1 %f, rouge2 %f, rougeL %f. Saving to %s', rouge_results['1'], rouge_results['2'], rouge_results['l'], bestmodel_save_path)
      evaluator._saver.save(evaluator._sess, bestmodel_save_path, global_step=current_step, latest_filename='checkpoint_best')
      rouge_results['step'] = current_step
      with open(best_rouge_file, 'wb') as f:
        pk.dump(rouge_results, f)

    # flush the summary writer every so often
    if current_step % 100 == 0:
      summary_writer.flush()
