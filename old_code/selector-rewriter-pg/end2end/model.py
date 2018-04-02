import os
import sys
import time
import tensorflow as tf
import pdb

FLAGS = tf.app.flags.FLAGS

class SelectorRewriter(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, select_model, rewrite_model):
    self._hps = hps
    self._selector = select_model
    self._rewriter = rewrite_model

  def _add_inconsistent_loss(self):
    hps = self._hps
    enc_len = tf.shape(self._rewriter._enc_sent_id_mask)[1]

    batch_nums = tf.expand_dims(tf.range(0, limit=hps.batch_size), 1) # shape (batch_size, 1)
    indices = tf.stack((tf.tile(batch_nums, [1, enc_len]), self._rewriter._enc_sent_id_mask), axis=2) # shape (batch_size, enc_len, 2)
    # All pad tokens will get probability of 0.0 since the sentence id is -1 (gather_nd will produce 0.0 for invalid indices)
    selector_probs_projected = tf.gather_nd(self._selector.probs, indices) # shape (batch_size, enc_len)

    '''
    if hps.inconsistent_thres == 'avg':
      prob_thres = tf.reduce_mean(tf.reduce_sum(self._selector.probs, 1) / tf.reduce_sum(self._selector._art_padding_mask, 1))
    else:
      prob_thres = hps.inconsistent_thres
    '''

    losses = []
    batch_nums_tilek = tf.tile(batch_nums, [1, hps.inconsistent_topk]) # shape (batch_size, k)
    # To compute inconsistent loss = -log((1 - sent_prob)*(1 - word_attn))
    for dec_step, attn_dist in enumerate(self._rewriter.attn_dists_norescale):
      '''
      max_w_id = tf.expand_dims(tf.argmax(attn_dist, axis=1), 1)  # shape (batch_size, 1)
      max_w_id = tf.cast(max_w_id, tf.int32)  # shape (batch_size, 1)
      max_w_indices = tf.concat((batch_nums, max_w_id), axis=1)  # shape (batch_size, 2)
      max_w = tf.gather_nd(attn_dist, max_w_indices)  # shape (batch_size,)
      s = tf.gather_nd(selector_probs_projected, max_w_indices)  # shape (batch_size,)
      loss_one_step = max_w * s  # shape (batch_size,)
      loss_one_step = -tf.log(loss_one_step + sys.float_info.epsilon)  # shape (batch_size,)
      '''
      topk_w, topk_w_id = tf.nn.top_k(attn_dist, hps.inconsistent_topk)  # shape (batch_size, topk)
      topk_w_indices = tf.stack((batch_nums_tilek, topk_w_id), axis=2)  # shape (batch_size, topk, 2)
      topk_s = tf.gather_nd(selector_probs_projected, topk_w_indices)  # shape (batch_size, topk)
      '''
      # log first than mean
      loss_one_step = -tf.log((topk_w * topk_s) + sys.float_info.epsilon)  # shape (batch_size, topk)
      loss_one_step = tf.reduce_mean(loss_one_step, 1)  # shape (batch_size,)
      '''
      # mean first than log
      loss_one_step = tf.reduce_mean(topk_w * topk_s, 1)  # shape (batch_size,)
      loss_one_step = -tf.log(loss_one_step + sys.float_info.epsilon)  # shape (batch_size,)


      loss_one_step *= self._rewriter._dec_padding_mask[:,dec_step]  # shape (batch_size,)
      losses.append(loss_one_step)
    loss = tf.reduce_mean(sum(losses)/tf.reduce_sum(self._rewriter._dec_padding_mask, axis=1))
    return loss
        

  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    hps = self._hps
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._rewriter._total_loss if hps.coverage else self._rewriter._loss
    if hps.selector_loss_in_end2end:
      if hps.loss == 'PG' and (hps.regu_ratio_wt > 0.0 or hps.regu_l2_wt > 0.0):
        loss_to_minimize += (self._selector._total_loss * hps.selector_loss_wt)
      else:
        loss_to_minimize += (self._selector._loss * hps.selector_loss_wt)

    if hps.inconsistent_loss:
      self._inconsistent_loss = self._add_inconsistent_loss()
      tf.summary.scalar('inconsist_loss', self._inconsistent_loss)
      loss_to_minimize += self._inconsistent_loss

    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device("/gpu:0"):
      grads, global_norm = tf.clip_by_global_norm(gradients, hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    if hps.optim == 'adagrad':
      tf.logging.info('Using Adagrad optimizer')
      optimizer = tf.train.AdagradOptimizer(hps.lr, initial_accumulator_value=hps.adagrad_init_acc)
    elif hps.optim == 'adam':
      tf.logging.info('Using Adam optimizer')
      optimizer = tf.train.AdamOptimizer()
    with tf.device("/gpu:0"):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    if self._hps.selector_loss_in_end2end and self._hps.loss == 'PG':
      self._selector.running_avg_reward = tf.Variable(0.0, name='running_avg_reward', trainable=False)

    self._selector._add_placeholders()
    self._rewriter._add_placeholders()
    with tf.device("/gpu:0"):
      self._selector._add_sent_selector()
      self._rewriter._add_seq2seq(selector_probs=self._selector.probs)
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op,
       summaries, loss, global_step and (optionally) coverage loss."""
    hps = self._hps
    feed_dict = self._selector._make_feed_dict(batch)

    if hps.selector_loss_in_end2end and hps.loss == 'PG':
      #(actions_argmax, actions_sample) = sess.run([self.actions_argmax, \
      #                                             self.actions_sample], feed_dict)  # (batch_size, max_art_len)
      #results_argmax = self._get_rewards(actions_argmax, batch)
      actions_sample = sess.run(self._selector.actions_sample, feed_dict)  # (batch_size, max_art_len)
      results_sample = self._selector._get_rewards(actions_sample, batch)
      feed_dict[self._selector.actions_sample] = results_sample['actions']
      feed_dict[self._selector.rewards] = results_sample['rewards']

    feed_dict.update(self._rewriter._make_feed_dict(batch))

    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'p_gen_avg': self._rewriter.p_gen_avg,
        'probs': self._selector.probs,
        'loss': self._rewriter._loss,
        'global_step': self.global_step,
    }
    if hps.coverage:
      to_return['coverage_loss'] = self._rewriter._coverage_loss
    if hps.inconsistent_loss:
      to_return['inconsist_loss'] = self._inconsistent_loss
    if hps.selector_loss_in_end2end:
      to_return['selector_loss'] = self._selector._loss
      if hps.loss == 'PG':
        if hps.regu_ratio_wt > 0.0:
          to_return['ratio_loss'] = self._selector._ratio_loss
          to_return['total_loss'] = self._selector._total_loss
        to_return['running_avg_reward'] = self._selector.assign_op  # new running avg reward after the asign op
        results = sess.run(to_return, feed_dict)
        #results['argmax'] = results_argmax
        results['sample'] = results_sample
        return results

    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries,
       loss, global_step and (optionally) coverage loss."""
    hps = self._hps
    feed_dict = self._selector._make_feed_dict(batch)
    if hps.selector_loss_in_end2end and hps.loss == 'PG':
      (actions_argmax, actions_sample) = sess.run([self._selector.actions_argmax, \
                                                   self._selector.actions_sample], feed_dict)  # (batch_size, max_art_len)
      results_argmax = self._selector._get_rewards(actions_argmax, batch)
      results_sample = self._selector._get_rewards(actions_sample, batch)
      feed_dict[self._selector.actions_sample] = results_sample['actions']
      feed_dict[self._selector.rewards] = results_sample['rewards']

    feed_dict.update(self._rewriter._make_feed_dict(batch))

    to_return = {
        'summaries': self._summaries,
        'p_gen_avg': self._rewriter.p_gen_avg,
        'probs': self._selector.probs,
        'loss': self._rewriter._loss,
        'global_step': self.global_step,
    }
    if hps.coverage:
      to_return['coverage_loss'] = self._rewriter._coverage_loss

    if hps.inconsistent_loss:
      to_return['inconsist_loss'] = self._inconsistent_loss

    if hps.selector_loss_in_end2end:
      to_return['selector_loss'] = self._selector._loss
      if hps.loss == 'PG':
        if hps.regu_ratio_wt > 0.0:
          to_return['ratio_loss'] = self._selector._ratio_loss
          to_return['total_loss'] = self._selector._total_loss
        results = sess.run(to_return, feed_dict)
        results['argmax'] = results_argmax
        results['sample'] = results_sample
        #results['batch_avg_reward'] = batch_avg_reward
        #results['batch_avg_ratio'] = batch_avg_ratio
        #results['batch_avg_gt_recall'] = batch_avg_gt_recall
        return results

    return sess.run(to_return, feed_dict)


  def run_greedy_search(self, sess, batch):
    hps = self._hps
    feed_dict = self._selector._make_feed_dict(batch)
    feed_dict.update(self._rewriter._make_feed_dict(batch))
    return sess.run(self._rewriter.greedy_search_words, feed_dict)
