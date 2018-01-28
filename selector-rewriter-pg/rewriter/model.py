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

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder_one_step
from tensorflow.contrib.tensorboard.plugins import projector
import data
import pdb

FLAGS = tf.app.flags.FLAGS

class Rewriter(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab
    if hps.mode == 'train':
      self._graph_mode = 'teacher_forcing'
    elif hps.mode == 'eval':
      if hps.eval_method == 'loss':
        self._graph_mode = 'teacher_forcing'
      elif hps.eval_method == 'rouge':
        self._graph_mode = hps.decode_method + '_search'
    elif hps.mode == 'evalall':
      self._graph_mode = hps.decode_method + '_search'

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    # encoder part
    self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
    self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
    self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
    if hps.model == 'end2end':
      self._enc_sent_id_mask = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_sent_id_mask')
    if FLAGS.pointer_gen:
      self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
      self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

    # decoder part
    if self._graph_mode in ['teacher_forcing', 'beam_search']:
      self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
      self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
      self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

      if self._graph_mode == 'beam_search':
        self.prev_context = tf.placeholder(tf.float32, [hps.batch_size, hps.hidden_dim_rewriter*2], name='prev_context')
        if hps.coverage:
          self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')


  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    hps = self._hps
    feed_dict = {}
    feed_dict[self._enc_batch] = batch.enc_batch
    feed_dict[self._enc_lens] = batch.enc_lens
    feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
    if hps.model == 'end2end':
      feed_dict[self._enc_sent_id_mask] = batch.enc_sent_id_mask
    if FLAGS.pointer_gen:
      feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed_dict[self._max_art_oovs] = batch.max_art_oovs
    if just_enc == False:
      if self._graph_mode in ['teacher_forcing', 'beam_search']:
        feed_dict[self._dec_batch] = batch.dec_batch
        feed_dict[self._target_batch] = batch.target_batch_rewriter
        feed_dict[self._dec_padding_mask] = batch.dec_padding_mask

    return feed_dict

  def _add_encoder(self, encoder_inputs, seq_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder'):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim_rewriter, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim_rewriter, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs, fw_st, bw_st


  def _reduce_states(self, fw_st, bw_st):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim_rewriter
    with tf.variable_scope('reduce_final_st'):

      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

      # Apply linear layer
      old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
      old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
      new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
      return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state


  def _add_decoder_one_step(self, inputs):
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In evalall (beam search) mode, you call this once for EACH decoder step.

    Args:
      inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

    Returns:
      outputs: List of tensors; the outputs of the decoder
      out_state: The final state of the decoder
      attn_dists: A list of tensors; the attention distributions
      p_gens: A list of scalar tensors; the generation probabilities
      coverage: A tensor, the current coverage vector
    """
    hps = self._hps
    cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim_rewriter, state_is_tuple=True, \
                                   initializer=self.rand_unif_init, \
                                   reuse=tf.get_variable_scope().reuse)

    if self._graph_mode == 'beam_search':
      # In evalall mode, our graph only contain one step deocder.
      # So need to feed in the previous step's context and coverage vector through placeholders.
      prev_coverage = self.prev_coverage if hps.coverage else None 
      prev_context = self.prev_context 
    else:
      prev_coverage = self.coverage if hps.coverage else None
      prev_context = self.context_vector

    if hps.model == 'end2end':
      selector_probs = self._selector_probs
      enc_sent_id_mask = self._enc_sent_id_mask
    else:
      selector_probs = None
      enc_sent_id_mask = None

    output, out_state, attn_dist_norescale, attn_dist, context_vector, p_gen, coverage = attention_decoder_one_step(\
                                                inputs, self._dec_out_state, self._enc_states, \
                                                self._enc_padding_mask, cell, \
                                                prev_context=prev_context, \
                                                pointer_gen=hps.pointer_gen, \
                                                use_coverage=hps.coverage, \
                                                prev_coverage=prev_coverage, \
                                                selector_probs=selector_probs, \
                                                enc_sent_id_mask=enc_sent_id_mask)

    return output, out_state, attn_dist_norescale, attn_dist, context_vector, p_gen, coverage

  def _add_decoder(self, dec_inputs, reuse=False):
    '''Add decoder with max_dec_steps for train and eval mode, 1 step for decode mode.'''
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary
    dec_steps = hps.max_dec_steps

    with tf.variable_scope('decoder'):
      attn_dists_norescale = []
      attn_dists = []
      p_gens = []
      vocab_scores = []
      log_dists = []

      if self._graph_mode == 'greedy_search':
        predict_words = []

      for i in range(dec_steps):
        tf.logging.info("Adding attention_decoder TF timestep %i of %i", i, dec_steps)
        #################################################
        # prepare decoder input for this step           #
        #################################################
        if i == 0:
          if reuse: tf.get_variable_scope().reuse_variables() # reuse variables
          self._dec_out_state = self._dec_in_state # initial the decoder state
          if self._graph_mode in ['teacher_forcing', 'greedy_search']:
            # For train and eval mode, this is the first step, 
            # initialize context vector and coverage vector
            self.context_vector = None
            if hps.coverage: self.coverage = None
            inp = tf.tile(tf.constant([self._vocab.word2id(data.START_DECODING)]), [hps.batch_size]) # shape (batch_size,)
          else:  # decode mode and beam search
            inp = dec_inputs[0]
        else:
          tf.get_variable_scope().reuse_variables() # reuse variables
          if self._graph_mode == 'teacher_forcing':
            inp = dec_inputs[i] # shape (batch_size,)
          else:  # decode mode and greedy search
            assert hps.decode_method == 'greedy'
            inp = predict_word # shape (batch_size,)
            # check if input is an OOV, if yes, replace with UNK token
            unk_batch = tf.tile(tf.constant([self._vocab.word2id(data.UNKNOWN_TOKEN)]), [hps.batch_size]) # shape (batch_size,)
            inp = tf.where(tf.less(inp, tf.constant(vsize)), inp, unk_batch) # shape (batch_size,)

        #################################
        # run one decoder step          #
        #################################
        inp_emb = tf.nn.embedding_lookup(self.embedding, inp) # (batch_size, emb_size)
        decoder_output, self._dec_out_state, attn_dist_norescale, attn_dist, self.context_vector, p_gen, self.coverage = \
                                                    self._add_decoder_one_step(inp_emb)

        attn_dists_norescale.append(attn_dist_norescale)
        attn_dists.append(attn_dist) # coverage loss need this
        p_gens.append(p_gen) # for summary

        #####################################################################
        # Add the output projection to obtain the vocabulary distribution   #
        #####################################################################
        with tf.variable_scope('output_projection'):
          w = tf.get_variable('w', [hps.hidden_dim_rewriter, vsize], dtype=tf.float32, \
                                                initializer=self.trunc_norm_init)
          w_t = tf.transpose(w)
          v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
          vocab_score = (tf.nn.xw_plus_b(decoder_output, w, v)) # apply the linear layer
          vocab_scores.append(vocab_score) # baseline model (no pointer_gen) need this
          vocab_dist = tf.nn.softmax(vocab_score)

        ##############################################################################
        # For pointer-generator model, calc final dist from copy dist and vocab dist #
        ##############################################################################
        if FLAGS.pointer_gen:
          final_dist = self._calc_final_dist_one_step(vocab_dist, attn_dist, p_gen)
          log_dist = tf.log(final_dist)
        else: # final distribution is just vocabulary distribution
          log_dist = tf.log(vocab_dist)
        log_dists.append(log_dist)

        if self._graph_mode == 'greedy_search':
          predict_word = tf.stop_gradient(tf.to_int32(tf.argmax(log_dist, 1))) # shape (batch_size,)
          predict_words.append(predict_word)

    #####################################################
    # Return the values that are needed in current mode #
    #####################################################
    if self._graph_mode == 'greedy_search':
      return tf.stack(predict_words, axis=1)  # shape (batch_size, max_dec_steps)
    else:
      return vocab_scores, log_dists, attn_dists_norescale, attn_dists, p_gens


  def _calc_final_dist_one_step(self, vocab_dist, attn_dist, p_gen):
    with tf.variable_scope('final_distribution'):
      batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
      batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
      attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
      batch_nums_tile = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)

      '''
      # If end2end, multiply the selector sentence probability with attnention probability
      if self._hps.model == 'end2end':
        indices = tf.stack( (batch_nums_tile, self._enc_sent_id_mask), axis=2) # shape (batch_size, attn_len, 2)
        selector_probs_projected = tf.gather_nd(self._selector_probs, indices)
        attn_dist = attn_dist * selector_probs_projected
        attn_dist = attn_dist / tf.reduce_sum(attn_dist, axis=1, keep_dims=True)  # normalize the attnetion distribution
      '''

      # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
      vocab_dist = p_gen * vocab_dist
      attn_dist = (1-p_gen) * attn_dist

      # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
      extended_vsize = self._vocab.size() + self._max_art_oovs # the maximum (over the batch) size of the extended vocabulary
      extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
      vocab_dist_extended = tf.concat(axis=1, values=[vocab_dist, extra_zeros])# shape (batch_size, extended_vsize)

      # Project the values in the attention distributions onto the appropriate entries in the final distributions
      # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
      # This is done for each decoder timestep.
      # This is fiddly; we use tf.scatter_nd to do the projection
      indices = tf.stack( (batch_nums_tile, self._enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
      shape = [self._hps.batch_size, extended_vsize]
      attn_dist_projected = tf.scatter_nd(indices, attn_dist, shape) # shape (batch_size, extended_vsize)

      # Add the vocab distributions and the copy distributions together to get the final distributions
      # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
      # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
      final_dist = vocab_dist_extended + attn_dist_projected

      # OOV part of vocab is max_art_oov long. Not all the sequences in a batch will have max_art_oov tokens.
      # That will cause some entries to be 0 in the distribution, which will result in NaN when calulating log_dists
      # Add a very small number to prevent that.
      def add_epsilon(dist, epsilon=sys.float_info.epsilon):
        epsilon_mask = tf.ones_like(dist) * epsilon
        return dist + epsilon_mask

      final_dist = add_epsilon(final_dist)

      return final_dist


  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)

  def _add_seq2seq(self, selector_probs=None):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary

    with tf.variable_scope('seq2seq'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

      ####################################################################
      # Add embedding matrix (shared by the encoder and decoder inputs)  #
      ####################################################################
      with tf.variable_scope('embedding'):
        self.embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, \
                                          initializer=self.trunc_norm_init)
        if hps.mode=="train": self._add_emb_vis(self.embedding) # add to tensorboard
        emb_enc_inputs = tf.nn.embedding_lookup(self.embedding, self._enc_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        if self._graph_mode in ['teacher_forcing', 'beam_search']:
          dec_inputs = tf.unstack(self._dec_batch, axis=1) # list of max_dec_steps tensor with shape (batch_size,)

      ########################################
      # Add the encoder.                     #
      ########################################
      enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
      self._enc_states = enc_outputs

      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
      self._dec_in_state = self._reduce_states(fw_st, bw_st)

      ########################################
      # Add the decoder.                     #
      ########################################
      if hps.model == 'end2end': 
        self._selector_probs = selector_probs

      if self._graph_mode == 'greedy_search':
        self.greedy_search_words = self._add_decoder(dec_inputs=None)
      else:
        vocab_scores, log_dists, self.attn_dists_norescale, self.attn_dists, self.p_gens = self._add_decoder(dec_inputs)

      ################################################
      # Calculate the loss for train and eval mode   #
      ################################################
      if self._graph_mode == 'teacher_forcing':
        with tf.variable_scope('loss'):
          ################################
          # teacher forcing loss         #
          ################################
          if FLAGS.pointer_gen: 
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the log probs of the gold target words
            loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
            batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
            for dec_step, log_dist in enumerate(log_dists):
              targets = self._target_batch[:,dec_step] # indices of target words. shape (batch_size)
              indices = tf.stack((batch_nums, targets), axis=1) # shape (batch_size, 2)
              losses = tf.gather_nd(-log_dist, indices) # shape (batch_size). loss on this step for each batch
              loss_per_step.append(losses)

            # Apply dec_padding_mask and get loss
            self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)
          else: # baseline model
            # this function applies softmax internally
            self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), \
                                                  self._target_batch, self._dec_padding_mask) 

          tf.summary.scalar('loss', self._loss)
          self.p_gen_avg = tf.reduce_mean(tf.stack(self.p_gens))
          tf.summary.scalar('p_gen', self.p_gen_avg)

          ##############################################################
          # Calculate coverage loss from the attention distributions   #
          ##############################################################
          if hps.coverage:
            with tf.variable_scope('coverage_loss'):
              self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
              tf.summary.scalar('coverage_loss', self._coverage_loss)
            self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
            tf.summary.scalar('total_loss', self._total_loss)

    ##########################################
    # produce top K words in decode mode     #
    ##########################################
    if self._graph_mode == 'beam_search':
      # We run decode beam search mode one decoder step at a time
      assert len(log_dists)==1 # log_dists is a singleton list containing shape (batch_size, extended_vsize)
      log_dists = log_dists[0]
      self._topk_log_probs, self._topk_ids = tf.nn.top_k(log_dists, hps.batch_size*2) # note batch_size=beam_size in decode mode


  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device("/gpu:0"):
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
    with tf.device("/gpu:0"):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device("/gpu:0"):
      #if self._hps.model == 'end2end':
      #  self._selector_probs = selector_probs
      self._add_seq2seq()
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
    feed_dict = self._make_feed_dict(batch)

    to_return = {
        'train_op': self._train_op,
        'p_gen_avg': self.p_gen_avg,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss

    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, 
       loss, global_step and (optionally) coverage loss."""
    hps = self._hps
    feed_dict = self._make_feed_dict(batch)

    to_return = {
        'summaries': self._summaries,
        'p_gen_avg': self.p_gen_avg,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss

    return sess.run(to_return, feed_dict)

  def run_greedy_search(self, sess, batch):
    hps = self._hps
    feed_dict = self._make_feed_dict(batch)
    return sess.run(self.greedy_search_words, feed_dict)


  def run_encoder(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
    #(enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder
    (enc_states, dec_in_state) = sess.run([self._enc_states, self._dec_in_state], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return enc_states, dec_in_state


  def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, \
                     prev_context, prev_coverage, selector_probs=None):
    """For beam search decoding. Run the decoder for one step.

    Args:
      sess: Tensorflow session.
      batch: Batch object containing single example repeated across the batch
      latest_tokens: Tokens to be fed as input into the decoder for this timestep
      enc_states: The encoder states.
      dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
      prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

    Returns:
      ids: top 2k ids. shape [beam_size, 2*beam_size]
      probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
      new_states: new states of the decoder. a list length beam_size containing
        LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
      attn_dists: List length beam_size containing lists length attn_length.
      p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
      new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
    """

    beam_size = len(dec_init_states)

    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    feed = {
        self._enc_states: enc_states,
        self._enc_padding_mask: batch.enc_padding_mask,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens])),
        self.prev_context: np.stack(prev_context, axis=0)
    }

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "attn_dists": self.attn_dists,
      "context_vector": self.context_vector
    }

    if FLAGS.pointer_gen:
      feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed[self._max_art_oovs] = batch.max_art_oovs
      to_return['p_gens'] = self.p_gens

    if self._hps.coverage:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      to_return['coverage'] = self.coverage

    if self._hps.model == 'end2end':
      feed[self._selector_probs] = selector_probs
      feed[self._enc_sent_id_mask] = batch.enc_sent_id_mask
      to_return['attn_dists_norescale'] = self.attn_dists_norescale

    results = sess.run(to_return, feed_dict=feed) # run the decoder step

    # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in xrange(beam_size)]

    # Convert a tensor to a list of k arrays
    attn_dists = results['attn_dists'][0].tolist()
    new_context = results['context_vector'].tolist()

    if FLAGS.pointer_gen:
      # Convert a tensor to a list of k arrays
      p_gens = results['p_gens'][0].tolist()
    else:
      p_gens = [None for _ in xrange(beam_size)]

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      new_coverage = results['coverage'].tolist()
      assert len(new_coverage) == beam_size
    else:
      new_coverage = [None for _ in xrange(beam_size)]

    if self._hps.model == 'end2end':
      attn_dists_norescale = results['attn_dists_norescale'][0].tolist()
    else:
      attn_dists_norescale = None

    return results['ids'], results['probs'], new_states, attn_dists_norescale, attn_dists, new_context, p_gens, new_coverage


def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """
  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average


def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. 
                A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss


