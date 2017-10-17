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
import data
import reward_criterion
from attention_decoder import attention_decoder_one_step
from tensorflow.contrib.tensorboard.plugins import projector
#from nltk import ngrams
import multiprocessing
#from joblib import Parallel, delayed
import pdb

FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab
    #self.current_ss = 0.

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    # encoder part
    self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
    self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
    self._cap_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='cap_batch')
    self._cap_lens = tf.placeholder(tf.int32, [hps.batch_size], name='cap_lens')
    self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
    self._cap_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='cap_padding_mask')
    if FLAGS.pointer_gen:
      self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
      self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

    # decoder part
    self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
    self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

    if hps.mode=="decode":
      self.prev_context = tf.placeholder(tf.float32, [hps.batch_size, hps.hidden_dim*2], name='prev_context')
      self.prev_context_cap = tf.placeholder(tf.float32, [hps.batch_size, hps.hidden_dim*2], name='prev_context_cap')
      if hps.coverage:
        self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')
        self.prev_coverage_cap = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage_cap')
    # reward
    if hps.training_method == 'PG':
      #self.is_sample = tf.placeholder(tf.bool, [], name='is_sample')
      self.feed_actions = tf.placeholder(tf.bool, [], name='feed_actions')
      # advantage = reward - baseline
      self.advantages = tf.placeholder("float32", [hps.batch_size * hps.max_dec_steps], name='advantages')
      self.rewards = tf.placeholder("float32", [hps.batch_size * hps.max_dec_steps], name='rewards')


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
    feed_dict[self._cap_batch] = batch.cap_batch
    feed_dict[self._cap_lens] = batch.cap_lens
    feed_dict[self._cap_padding_mask] = batch.cap_padding_mask    
    if FLAGS.pointer_gen:
      feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed_dict[self._max_art_oovs] = batch.max_art_oovs
    if just_enc == False and hps.training_method == 'TF':
      feed_dict[self._dec_batch] = batch.dec_batch
      feed_dict[self._target_batch] = batch.target_batch
      feed_dict[self._dec_padding_mask] = batch.dec_padding_mask

    return feed_dict

  def _add_encoder(self, encoder_inputs, seq_len, name):
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
    with tf.variable_scope(name):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs, fw_st, bw_st


  def _reduce_states(self, fw_st, bw_st, name):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim
    with tf.variable_scope(name):

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
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

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
    cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, \
                                   initializer=self.rand_unif_init, \
                                   reuse=tf.get_variable_scope().reuse)

    if hps.mode == 'decode':
      # In decode mode, our graph only contain one step deocder.
      # So need to feed in the previous step's context and coverage vector through placeholders.
      prev_coverage = self.prev_coverage if hps.coverage else None
      prev_coverage_cap = self.prev_coverage_cap if hps.coverage else None 
      prev_context = self.prev_context 
      prev_context_cap = self.prev_context_cap
    else:
      prev_coverage = self.coverage if hps.coverage else None
      prev_coverage_cap = self.prev_coverage_cap if hps.coverage else None
      prev_context = self.context_vector
      prev_context_cap = self.context_vector_cap
    output, out_state, attn_dist, context_vector, coverage, attn_dist_cap, context_vector_cap, coverage_cap, p_gen = attention_decoder_one_step(\
                                                inputs, self._dec_out_state, self._enc_states, \
                                                self._cap_states, \
                                                self._enc_padding_mask, self._cap_padding_mask, cell, \
                                                prev_context=prev_context, prev_context_cap=prev_context_cap, \
                                                pointer_gen=hps.pointer_gen, \
                                                use_coverage=hps.coverage, \
                                                prev_coverage=prev_coverage, prev_coverage_cap=prev_coverage_cap)

    return output, out_state, attn_dist, context_vector, coverage, attn_dist_cap, context_vector_cap, coverage_cap, p_gen

  def _add_decoder(self, dec_inputs, predict_method='sample', reuse=False):
    '''Add decoder with max_dec_steps for train and eval mode, 1 step for decode mode.'''
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary
    dec_steps = hps.max_dec_steps if hps.mode != 'decode' else 1 

    with tf.variable_scope('decoder'):
      attn_dists = []
      attn_dists_cap = []
      p_gens = []
      vocab_scores = []
      log_dists = []

      if hps.training_method == 'PG':
        predict_words = []

      for i in range(dec_steps):
        #################################################
        # prepare decoder input for this step           #
        #################################################
        if i == 0:
          if reuse: tf.get_variable_scope().reuse_variables() # reuse variables
          self._dec_out_state = self._dec_in_state # initial the decoder state
          if hps.mode != 'decode':
            # For train and eval mode, this is the first step, assign the BOS token as input
            # initialize context vector and coverage vector
            inp = tf.tile(tf.constant([self._vocab.word2id(data.START_DECODING)]), [hps.batch_size]) # shape (batch_size,)
            self.context_vector = None
            self.context_vector_cap = None
            if hps.coverage: 
              self.coverage = None
              self.coverage_cap = None
          else:
            # decode mode will only go here (run one step at a time)
            inp = dec_inputs[0] # shape (batch_size,)
        else:
          tf.get_variable_scope().reuse_variables() # reuse variables
          if hps.training_method == 'TF' and hps.mode == 'train': # teacher forcing
            inp = dec_inputs[i] # shape (batch_size,)
          else: # policy gradient
            inp = predict_word # shape (batch_size,)
            # check if input is an OOV, if yes, replace with UNK token
            unk_batch = tf.tile(tf.constant([self._vocab.word2id(data.UNKNOWN_TOKEN)]), [hps.batch_size]) # shape (batch_size,)
            inp = tf.where(tf.less(inp, tf.constant(vsize)), inp, unk_batch) # shape (batch_size,)

        #################################
        # run one decoder step          #
        #################################
        inp_emb = tf.nn.embedding_lookup(self.embedding, inp) # (batch_size, emb_size)
        decoder_output, self._dec_out_state, attn_dist, self.context_vector, self.coverage, attn_dist_cap, self.context_vector_cap, self.coverage_cap, p_gen = self._add_decoder_one_step(inp_emb)

        attn_dists.append(attn_dist) # coverage loss need this
        attn_dists_cap.append(attn_dist_cap)
        p_gens.append(p_gen) # for summary

        #####################################################################
        # Add the output projection to obtain the vocabulary distribution   #
        #####################################################################
        with tf.variable_scope('output_projection'):
          w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, \
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
          final_dist = self._calc_final_dist_one_step(vocab_dist, attn_dist, p_gen) # shape (batch_size, vocab_size)
          log_dist = tf.log(final_dist)
        else: # final distribution is just vocabulary distribution
          log_dist = tf.log(vocab_dist)
        log_dists.append(log_dist)

        ####################################################################
        # For policy gradient, predict a word or use the sampled action    #
        ####################################################################
        if hps.training_method == 'PG':
          if predict_method == 'argmax':
            tf.logging.info("Adding attention_decoder argmax timestep %i of %i", i, dec_steps)
            predict_word = tf.stop_gradient(tf.to_int32(tf.argmax(log_dist, 1))) # shape (batch_size,)
          elif predict_method == 'sample':
            tf.logging.info("Adding attention_decoder sample timestep %i of %i", i, dec_steps)
            predict_word = tf.squeeze(tf.stop_gradient(tf.to_int32(tf.multinomial(log_dist, 1)))) # shape (batch_size,)
            predict_word = tf.where(self.feed_actions, dec_inputs[i], predict_word)
          predict_words.append(predict_word)
        else:
          tf.logging.info("Adding attention_decoder TF timestep %i of %i", i, dec_steps)

    #####################################################
    # Return the values that are needed in current mode #
    #####################################################
    if hps.training_method == 'PG': # train and eval mode for policy gradient
      predict_words = tf.stack(predict_words, axis=1) # shape (batch_size, max_dec_steps)
      return predict_words, log_dists, attn_dists, attn_dists_cap, p_gens
    else: # train and eval mode for teacher forcing
      return vocab_scores, log_dists, attn_dists, attn_dists_cap, p_gens


  def _calc_final_dist_one_step(self, vocab_dist, attn_dist, p_gen):
    with tf.variable_scope('final_distribution'):
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
      batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
      batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
      attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
      batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
      indices = tf.stack( (batch_nums, self._enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
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

  def _add_seq2seq(self):
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
        emb_enc_inputs = tf.nn.embedding_lookup(self.embedding, self._enc_batch) # tensor with shap (batch_size, max_enc_steps, emb_size)
        emb_cap_inputs = tf.nn.embedding_lookup(self.embedding, self._cap_batch)
        # For teacher forcing, dec_inputs are gound truth
        # For policy gradient, dec_inputs are sampled actions (will be fed in the 2nd session run)
        dec_inputs = tf.unstack(self._dec_batch, axis=1) # list of max_dec_steps tensor with shape (batch_size,)

      ########################################
      # Add the encoder.                     #
      ########################################
      enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens, name='article_encoder')
      cap_outputs, _, _ = self._add_encoder(emb_cap_inputs, self._cap_lens, name='caption_encoder')
      self._enc_states = enc_outputs
      self._cap_states = cap_outputs

      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
      self._dec_in_state = self._reduce_states(fw_st, bw_st, 'reduce_art_final_st')

      ########################################
      # Add the decoder.                     #
      ########################################
      if hps.training_method == 'TF':
        vocab_scores, log_dists, self.attn_dists, self.attn_dists_cap, self.p_gens = self._add_decoder(dec_inputs)
      elif hps.training_method == 'PG':
        self.pred_words_sample, log_dists, self.attn_dists, self.attn_dists_cap, self.p_gens = self._add_decoder(dec_inputs)
        self.pred_words_argmax, _, _, _, _ = self._add_decoder(None, predict_method='argmax', reuse=True)

      #self.action_log_probs = action_log_probs_argmax

      ################################################
      # Calculate the loss for train and eval mode   #
      ################################################
      if hps.mode in ['train', 'eval']:
        with tf.variable_scope('loss'):
          if hps.training_method == 'TF': 
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
          else:
            ################################# 
            # policy gradient loss          #
            #################################
            action_log_probs = []
            predict_masks = []
            mask = tf.constant(True, "bool", [hps.batch_size]) # shape (batch_size,)

            for i, log_dist in enumerate(log_dists):
              # mask out the word beyond <END>
              batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size,)
              indices = tf.stack((batch_nums, dec_inputs[i]), axis=1) # shape (batch_size, 2)
              action_log_prob = tf.gather_nd(log_dist, indices) * tf.to_float(mask) # shape (batch_size,)
              action_log_probs.append(action_log_prob)
              predict_masks.append(tf.to_float(mask))

              # create next mask
              end_id = tf.constant(self._vocab.word2id(data.STOP_DECODING), dtype=tf.int32)
              is_end = tf.not_equal(dec_inputs[i], end_id) # shape (batch_size,)
              mask = tf.logical_and(mask, is_end) # new mask for next step, shape (batch_size,)

            # Combine all actions' probabilities to a single tensor, so does the masks
            action_log_probs = tf.stack(action_log_probs, axis=1) # shape (batch_size, max_dec_steps)
            action_log_probs_flat = tf.reshape(action_log_probs, [-1]) # shape (batch_size*max_dec_steps,)
            predict_masks = tf.stack(predict_masks, axis=1) # shape (batch_size, max_dec_steps)
            predict_masks_flat = tf.reshape(predict_masks, [-1]) # shape (batch_size*max_dec_steps,)

            self._loss = -tf.reduce_sum(action_log_probs_flat * self.advantages) / tf.reduce_sum(predict_masks_flat)
            # add reward mean to tensorboard summary
            #reward_masked = self.rewards * predict_masks_flat # shape (batch_size * max_dec_step,)
            #self.reward_mean = tf.reduce_sum(reward_masked) / tf.reduce_sum(predict_masks_flat)
            self.reward_mean = tf.reduce_mean(self.rewards)
            tf.summary.scalar('reward', self.reward_mean)
            #baseline_masked = (self.rewards - self.advantages) * predict_masks_flat # shape (batch_size * max_dec_step,)
            #self.baseline_mean = tf.reduce_sum(baseline_masked) / tf.reduce_sum(predict_masks_flat)
            self.baseline_mean = tf.reduce_mean((self.rewards - self.advantages))
            tf.summary.scalar('baseline', self.baseline_mean)

          tf.summary.scalar('loss', self._loss)
          self.p_gen_avg = tf.reduce_mean(tf.stack(self.p_gens))
          tf.summary.scalar('p_gen', self.p_gen_avg)

          ##############################################################
          # Calculate coverage loss from the attention distributions   #
          ##############################################################
          if hps.coverage:
            with tf.variable_scope('coverage_loss'):
              if hps.training_method == 'TF':
                self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
                self._coverage_loss += _coverage_loss(self.attn_dists_cap, self._dec_padding_mask)
              else:
                self._coverage_loss = _coverage_loss(self.attn_dists, predict_masks)
                self._coverage_loss += _coverage_loss(self.attn_dists_cap, predict_masks)
              tf.summary.scalar('coverage_loss', self._coverage_loss)
            self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
            tf.summary.scalar('total_loss', self._total_loss)

    ##########################################
    # produce top K words in decode mode     #
    ##########################################
    if hps.mode == "decode":
      # We run decode beam search mode one decoder step at a time
      # For policy gradient, results will be the same for running sample decoder or argmax decoder,
      # because we will only run one decoder step and will use neither self.pred_words_sample nor self.pred_words_argmax
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
      self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)


  def _get_actions_rewards(self, sess, feed_dict, batch):
    '''Sample actions and get reward for policy gradient training.'''
    hps = self._hps
    feed_dict[self.feed_actions] = False
    feed_dict[self._dec_batch] = np.zeros((hps.batch_size, hps.max_dec_steps))
    
    t0=time.time()
    if hps.use_baseline:
      pred_words_argmax, pred_words_sample = sess.run([self.pred_words_argmax, self.pred_words_sample], feed_dict)  # B, S
      pred_words_argmax[:,-1] = self._vocab.word2id(data.STOP_DECODING)
      pred_words_sample[:,-1] = self._vocab.word2id(data.STOP_DECODING)
    else:
      pred_words_sample = sess.run(self.pred_words_sample, feed_dict)  # B, S
      pred_words_sample[:,-1] = self._vocab.word2id(data.STOP_DECODING)
    t1=time.time()
    tf.logging.info('seconds for predict actions: %.3f', t1-t0)

    advantages = []
    rewards = []
    t0=time.time()
    for i in range(hps.batch_size):
      # Get rewards
      gen_sents, score_sample = _summary_score(self._vocab, pred_words_sample[i], \
                                      batch.original_abstracts_sents[i], \
                                      batch.original_articles[i], batch.art_oovs[i])
      rewards.append(score_sample)
      if i == 0:
        tf.logging.info('reference summary: %s', ' '.join(batch.original_abstracts_sents[i]))
        tf.logging.info('sample summary: %s', ' '.join(gen_sents))

      # Get advantages
      if hps.use_baseline:
        gen_sents, score_argmax = _summary_score(self._vocab, pred_words_argmax[i], \
                                      batch.original_abstracts_sents[i], \
                                      batch.original_articles[i], batch.art_oovs[i])
        advantages.append(score_sample - score_argmax) # len batch_size
        if i == 0:
          tf.logging.info('argmax summary: %s', ' '.join(gen_sents))
      else:
        advantages.append(score_sample)

    t1=time.time()
    tf.logging.info('seconds for calc rewards: %.3f', t1-t0)
   
    advantages = np.tile(np.array(advantages), hps.max_dec_steps)
    rewards = np.tile(np.array(rewards), hps.max_dec_steps)
 
    '''
    t0=time.time()
    inputs_argmax = [(self._vocab, words, ref, art, oovs) \
                     for words, ref, art, oovs in zip(pred_words_argmax.tolist(), \
                                                      batch.original_abstracts_sents, \
                                                      batch.original_articles, \
                                                      batch.art_oovs)]
    inputs_sample = [(self._vocab, words, ref, art, oovs) \
                     for words, ref, art, oovs in zip(pred_words_sample.tolist(), \
                                                      batch.original_abstracts_sents, \
                                                      batch.original_articles, \
                                                      batch.art_oovs)]
    inputs = inputs_argmax + inputs_sample

    #pred_words_list = np.concatenate((pred_words_argmax, pred_words_sample), axis=0)
    #pred_words_list = pred_words_list.tolist()
    #inputs = ((words, self._vocab, oovs) for words, oovs in zip(pred_words_list, batch.art_oovs))
    #r = Parallel(n_jobs=32)(map(delayed(self._summary_score), inputs))

    #def _summary_score_star(args):
    #  """Convert `f([1,2])` to `f(1,2)` call."""
    #  return self._summary_score(*args)

    pool = multiprocessing.Pool(processes=4)
    r = pool.map(_summary_score_star, inputs)
    t1=time.time()
    tf.logging.info('seconds for calc rewards (parallel): %.3f', t1-t0)
    '''
    return pred_words_sample, rewards, advantages


  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, 
       summaries, loss, global_step and (optionally) coverage loss."""
    hps = self._hps
    feed_dict = self._make_feed_dict(batch)

    # Get the reward for policy gradient
    if hps.training_method == 'PG':
      actions, rewards, advantages = self._get_actions_rewards(sess, feed_dict, batch)
      feed_dict[self.rewards] = rewards
      feed_dict[self.advantages] = advantages
      feed_dict[self._dec_batch] = actions
      feed_dict[self.feed_actions] = True

    to_return = {
        'train_op': self._train_op,
        'p_gen_avg': self.p_gen_avg,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if hps.training_method == 'PG':
      #to_return['pred_words'] = self.pred_words_sample # for checking if actions are fed successfully
      to_return['reward_mean'] = self.reward_mean
    if hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss

    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, 
       loss, global_step and (optionaly) coverage loss."""
    hps = self._hps
    feed_dict = self._make_feed_dict(batch)

    # Get the reward for policy gradient
    if hps.training_method == 'PG':
      actions, rewards, advantages = self._get_actions_rewards(sess, feed_dict, batch)
      feed_dict[self.rewards] = rewards
      feed_dict[self.advantages] = advantages
      feed_dict[self._dec_batch] = actions
      feed_dict[self.feed_actions] = True

    to_return = {
        'summaries': self._summaries,
        'p_gen_avg': self.p_gen_avg,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if hps.training_method == 'PG':
      to_return['reward_mean'] = self.reward_mean
    if hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss

    return sess.run(to_return, feed_dict)

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
    (enc_states, cap_states, dec_in_state, global_step) = sess.run([self._enc_states, self._cap_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return enc_states, cap_states, dec_in_state


  def decode_onestep(self, sess, batch, latest_tokens, enc_states, cap_states, dec_init_states, prev_context, prev_context_cap, prev_coverage, prev_coverage_cap):
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
        self._cap_states: cap_states,
        self._cap_padding_mask: batch.cap_padding_mask,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens])),
        self.prev_context: np.stack(prev_context, axis=0),
        self.prev_context_cap: np.stack(prev_context_cap, axis=0)
    }

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "attn_dists": self.attn_dists,
      "attn_dists_cap": self.attn_dists_cap,
      "context_vector": self.context_vector,
      "context_vector_cap": self.context_vector_cap
    }

    if FLAGS.pointer_gen:
      feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed[self._max_art_oovs] = batch.max_art_oovs
      to_return['p_gens'] = self.p_gens

    if self._hps.coverage:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      feed[self.prev_coverage_cap] = np.stack(prev_coverage_cap, axis=0)
      to_return['coverage'] = self.coverage
      to_return['coverage_cap'] = self.coverage_cap


    results = sess.run(to_return, feed_dict=feed) # run the decoder step

    # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in xrange(beam_size)]

    # Convert a tensor to a list of k arrays
    attn_dists = results['attn_dists'][0].tolist()
    attn_dists_cap = results['attn_dists_cap'][0].tolist()
    new_context = results['context_vector'].tolist()
    new_context_cap = results['context_vector_cap'].tolist()

    if FLAGS.pointer_gen:
      # Convert a tensor to a list of k arrays
      p_gens = results['p_gens'][0].tolist()
    else:
      p_gens = [None for _ in xrange(beam_size)]

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      new_coverage = results['coverage'].tolist()
      new_coverage_cap = results['coverage_cap'].tolist()
      assert len(new_coverage) == beam_size
    else:
      new_coverage = [None for _ in xrange(beam_size)]
      new_coverage_cap = [None for _ in xrange(beam_size)]

    return results['ids'], results['probs'], new_states, attn_dists, attn_dists_cap, \
           new_context, new_context_cap, p_gens, new_coverage, new_coverage_cap


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

def _summary_score(vocab, words_idx, ref_sents, article, art_oovs):
  words_str = data.outputids2words(words_idx, vocab, art_oovs) # list of word strings
  gen_sents = data.words2sents(words_str) # list of sentence strings
  #rouge_1, rouge_2, rouge_l = reward_criterion.rouge_scores(gen_sents, ref_sents)
  #result = reward_criterion.rouge_scores_quick(gen_sents, ref_sents, metric=[FLAGS.reward_metric])
  #copy_rate = reward_criterion.copy_rate(gen_sents, article)
  score = reward_criterion.reward(gen_sents, ref_sents, article)
  return gen_sents, score

def _summary_score_star(args):
  """Convert `f([1,2])` to `f(1,2)` call."""
  return _summary_score(*args)

'''
def _extract_loss(art_batch, abs_batch):
  # Calculate n-grams copy
  def get_ngrams(text, n):
    n_grams = ngrams(text, n)
    return [tf.squeeze(tf.stack(grams)) for grams in n_grams]  

  extract_loss = tf.constant(0)
  for i in range(FLAGS.batch_size):
    article = tf.split(art_batch[i], num_or_size_splits=FLAGS.max_enc_steps, axis=0)
    abstract = tf.split(abs_batch[i], num_or_size_splits=FLAGS.max_dec_steps, axis=0)
    #pdb.set_trace()
    for n in range(2, 11):
      ngrams_art = get_ngrams(article, n)
      ngrams_abs = get_ngrams(abstract, n)
      ngrams_loss = tf.constant(0)
      for g1 in ngrams_abs:
          for g2 in ngrams_art:
            is_equal = tf.equal(tf.reduce_sum(tf.to_float(g1) - tf.to_float(g2)), tf.constant(0.0))
            ngrams_loss = tf.cond(is_equal, lambda: tf.add(ngrams_loss, tf.constant(1)), \
                                            lambda: tf.identity(ngrams_loss))
      extract_loss = tf.add(extract_loss, ngrams_loss / len(ngrams_abs) * n)

  return extract_loss / FLAGS.batch_size
'''
