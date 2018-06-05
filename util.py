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

"""This file contains some utility functions"""

import tensorflow as tf
import time
import os
import numpy as np
import pdb
FLAGS = tf.app.flags.FLAGS

def get_config():
  """Returns config for tf.session"""
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  return config

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

def load_ckpt(saver, sess, ckpt_dir='train', ckpt_path=None):
  """Load checkpoint from the train directory and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
  ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
  while True:
    try:
      if not ckpt_path:
        latest_filename = "checkpoint_best" if "eval" in ckpt_dir else None
        ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
        ckpt_path = ckpt_state.model_checkpoint_path
      tf.logging.info('Loading checkpoint %s', ckpt_path)
      saver.restore(sess, ckpt_path)
      return ckpt_path
    except:
      tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
      time.sleep(10)


def get_select_accuracy_one_thres(article_sents, probs, gt_selected_ids, thres, \
                                  min_select=None, max_select=None, method='prob'):
  art_sent_num = len(article_sents)
  probs = probs[:art_sent_num] # remove the probabilities of padding sentences
  sorted_probs = np.sort(probs)[::-1]
  id_sort_by_prob = np.argsort(probs)[::-1]
  if method == 'prob':
    select_num = sum(sorted_probs > thres) # number of probabilities that is greater than thres
  elif method == 'ratio':
    select_num = int(round(float(len(article_sents))*thres))
  elif method == 'num':
    select_num = int(thres)
  else:
    raise Exception("Not available method: %s (should only be prob/ratio)" % method)

  if min_select and select_num < min_select:
    select_num = int(min_select)
  elif max_select and select_num > max_select:
    select_num = int(max_select)

  if select_num != 0:
    selected_ids = list(np.sort(id_sort_by_prob[:select_num]))
    selected_sents = [article_sents[i] for i in selected_ids]
  else:
    #if FLAGS.mode == 'eval_all':
    #  pdb.set_trace()
    selected_sents = []
    selected_ids = []
      
  TP = [idx for idx in selected_ids if idx in gt_selected_ids] # true positive
  TN = [idx for idx in range(art_sent_num) if idx not in selected_ids and idx not in gt_selected_ids] # true negative
  if select_num > 0:
    precision = float(len(TP)) / select_num
  else:
    precision = 0.0
  recall = float(len(TP)) / len(gt_selected_ids)
  accuracy = float(len(TP) + len(TN)) / art_sent_num

  if method in ['prob', 'num']:
    ratio = float(select_num) / art_sent_num # ratio of selected sentences and article
  elif method == 'ratio':
    ratio = thres

  return selected_sents, selected_ids, precision, recall, accuracy, ratio


def get_select_AP(article_sents, probs, gt_selected_ids, step=0.1, method='prob'):
  """method can be prob/ratio"""
  select_num = []
  precision = []
  recall = []
  accuracy = []
  ratios = []
  bins = np.arange(0.0, 1.0, step)
  for i in bins:
    _, ids, p, r, acc, ratio = get_select_accuracy_one_thres(article_sents, probs, \
                                                    gt_selected_ids, i, method=method)
    select_num.append(len(ids))
    precision.append(p)
    recall.append(r)
    accuracy.append(acc)
    ratios.append(ratio)
  #print 'precision:', precision
  #print 'recall:', recall
  #print 'avg precision:', sum(precision)/len(precision)
  #print 'avg recall:', sum(recall)/len(recall)
  avg_p = sum(precision)/len(precision)
  avg_r = sum(recall)/len(recall)
  avg_acc = sum(accuracy)/len(accuracy)
  return select_num, precision, recall, accuracy, ratios, avg_p, avg_r, avg_acc

def get_batch_precision_recall(batch_article_sents, batch_gt_ids, batch_probs, step=0.1, method='prob', tf_print=True):
  """Calculate the precision/recall for a batch"""
  batch_size = len(batch_article_sents)
  step_num = len(np.arange(0.0, 1.0, step))
  precisions = np.zeros((step_num))
  recalls = np.zeros((step_num))
  accuracys = np.zeros((step_num))
  ratios = np.zeros((step_num))
  sent_nums = np.zeros((step_num))
  avg_ps = []
  avg_rs = []
  avg_accs = []
  for i in range(batch_size):
    sent_num, ps, rs, accs, ratio, avg_p, avg_r, avg_acc = get_select_AP(batch_article_sents[i], \
                                                          batch_probs[i], \
                                                          batch_gt_ids[i], step, method)
    precisions += np.array(ps)
    recalls += np.array(rs)
    accuracys += np.array(accs)
    ratios += np.array(ratio)
    sent_nums += np.array(sent_num)
    avg_ps.append(avg_p)
    avg_rs.append(avg_r)
    avg_accs.append(avg_acc)

  sent_nums = sent_nums/float(batch_size)
  precisions = precisions/batch_size
  recalls = recalls/batch_size
  accuracys = accuracys/batch_size
  ratios = ratios/batch_size
  avg_ps = sum(avg_ps)/batch_size
  avg_rs = sum(avg_rs)/batch_size
  avg_accs = sum(avg_accs)/batch_size

  if tf_print:
    tf.logging.info("avg num of sentences: " + str(sent_nums))
    tf.logging.info('precision on all thres: ' + str(precisions))
    tf.logging.info('recalls on all thres: ' + str(recalls))
    tf.logging.info('accuracies on all thres: ' + str(accuracys))
    tf.logging.info('select ratios on all thres: ' + str(ratios))
    tf.logging.info('avg precision: %f', avg_ps)
    tf.logging.info('avg recall: %f', avg_rs)
    tf.logging.info('avg accuracy: %f', avg_accs)

  return sent_nums, precisions, recalls, accuracys, ratios, avg_ps, avg_rs, avg_accs

def get_batch_ratio(batch_article_sents, batch_gt_ids, batch_probs, target_recall=0.9, method='prob', tf_print=True):
  batch_size = len(batch_article_sents)
  max_recall = target_recall + 0.01
  min_recall = target_recall - 0.01

  # initial threshold
  if method == 'prob':
    thres = 0.1
  elif method == 'ratio':
    thres = 0.8

  min_thres = 0.0
  max_thres = 1.0
  recall = 0.0
  count = 0
  while (recall < min_recall or recall > max_recall) and count < 100:
    recalls = []
    ratios = []
    for i in range(batch_size):
      _, _, _, recall, _, ratio = get_select_accuracy_one_thres(batch_article_sents[i], batch_probs[i], \
                                                    batch_gt_ids[i], thres, method=method)
      recalls.append(recall)
      ratios.append(ratio)

    recall = sum(recalls) / float(batch_size)
    ratio = sum(ratios) / float(batch_size)

    if method == 'prob':
      if recall < min_recall:
        max_thres = thres
        thres -= ((thres - min_thres) / 2.0)
      elif recall > max_recall:
        min_thres = thres
        thres += ((max_thres - thres) / 2.0)
    elif method == 'ratio':
      if recall > max_recall:
        max_thres = thres
        thres -= ((thres - min_thres) / 2.0)
      elif recall < min_recall:
        min_thres = thres
        thres += ((max_thres - thres) / 2.0)
    count += 1
    #print count

  if recall < min_recall or recall > max_recall:
    if tf_print:
      tf.logging.warning('fail to reach target recall: '+str(target_recall))      
    recall = 0.0
    thres = 0.0
    ratio = 1.0

  if tf_print:
    tf.logging.info('recall: %f, ratio: %f, thres: %f', recall, ratio, thres)
  return recall, ratio, thres




