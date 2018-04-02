import os
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from highway import *
import copy 
import pdb


class TextClassifier():

    def __init__(self, sess, train_data, test_data, hps):
        self._sess = sess
        self._hps = hps
        self._train_data = train_data
        self._test_data = test_data
        self._vocab = self._train_data.vocab
        self._vocab_size = self._train_data.vocab_size
        self._start_id = self._vocab._start_id
        
    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        hps = self._hps

        # article part
        self._art_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='art_batch')
        self._art_lens = tf.placeholder(tf.int32, [hps.batch_size], name='art_lens')
        self._art_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='art_padding_mask')
      
        # abstract part
        self._ref_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='ref_batch')
        self._ref_lens = tf.placeholder(tf.int32, [hps.batch_size], name='ref_lens')
        self._ref_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='ref_padding_mask')
        
        self._ext_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='ext_batch')
        self._ext_lens = tf.placeholder(tf.int32, [hps.batch_size], name='ext_lens')
        self._ext_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='ext_padding_mask')

        self._wrong_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='wrong_batch')
        self._wrong_lens = tf.placeholder(tf.int32, [hps.batch_size], name='wrong_lens')
        self._wrong_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='wrong_padding_mask')

    def _make_feed_dict(self, batch):
        hps = self._hps
        feed_dict = {}
        feed_dict[self._art_batch] = batch['article']
        feed_dict[self._art_lens] = batch['art_len']

        feed_dict[self._ref_batch] = batch['ref']
        feed_dict[self._ref_lens] = batch['ref_len']
        
        feed_dict[self._ext_batch] = batch['ext']
        feed_dict[self._ext_lens] = batch['ext_len']

        feed_dict[self._wrong_batch] = batch['wrong']
        feed_dict[self._wrong_lens] = batch['wrong_len']
        return feed_dict


    def _cnn_encoder(self, text, is_article=True, name='cnn_encoder', reuse=False):
        hps = self._hps
        hidden_size = hps.hidden_size
        text_len = hps.max_article_len if is_article else hps.max_abstract_len
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)

        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.device('/gpu:0'), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self._vocab_size, hidden_size], "float32", random_uniform_init)
                embedded_chars = tf.nn.embedding_lookup(word_emb_W, text) # (batch_size, seq_len, hidden_size)
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)  # (batch_size, seq_len, hidden_size, 1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []

            for filter_size, num_filter in zip(hps.filter_sizes, hps.num_filters):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, hidden_size, 1, num_filter]
                    W = tf.get_variable("W", filter_shape, "float32", random_uniform_init)
                    b = tf.get_variable("b", [num_filter], "float32", random_uniform_init)
                    conv = tf.nn.conv2d(
                        embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, text_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            h_pool = tf.concat(pooled_outputs, 3)      # B,1,1,total filters
            h_pool_flat = tf.reshape(h_pool, [-1, hps.num_filters_total])        # b, total filters

            # Add highway
            with tf.variable_scope("highway"):
                h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)
            
        return h_highway

    def _lstm_encoder(self, text, length, is_article=True, name='lstm_encoder', reuse=False):
        hps = self._hps
        hidden_size = hps.hidden_size
        batch_size = hps.batch_size
        max_words = hps.max_article_len if is_article else hps.max_abstract_len
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)

        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.contrib.rnn.LSTMCell(hidden_size, state_is_tuple=True, reuse=reuse)
                lstm1 = tf.contrib.rnn.DropoutWrapper(lstm1, output_keep_prob=1-hps.drop_out_rate)
            with tf.device('/gpu:0'), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self._vocab_size, hidden_size], "float32", random_uniform_init)

            state = lstm1.zero_state(batch_size, 'float32')
            start_token = tf.constant(self._start_id, dtype=tf.int32, shape=[batch_size])
            # VQA use states
            state_list = []
            for j in range(max_words+1):
                if j > 0:
                    tf.get_variable_scope().reuse_variables()
                with tf.device('/gpu:0'):
                    if j ==0:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, start_token)
                    else:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, text[:,j-1])
                with tf.variable_scope("lstm"):
                    # "generator/lstm"
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())     # output: B,H
                # apppend state from index 1 (the start of the word)
                if j > 0:
                    state_list.append(tf.concat([state[0], state[1]], 1))

            state_list = tf.stack(state_list, 1)    # B,S,2H

            # length-1 => index start from 0
            # need to prevent length = 0
            length_index = length-1
            condition = tf.greater_equal(length_index, 0)       # B
            length_index = tf.where(condition, length_index, tf.constant(0, dtype=tf.int32, shape=[batch_size]))
            idx = tf.stack((tf.range(batch_size), length_index), 1)
            state_gather = tf.gather_nd(state_list, idx)        # B, 2H
        return state_gather


    def _text_classifier(self, article_feature, abstract_feature, name="text_classifier", reuse=False):
        hps = self._hps
        if hps.encode_method == 'lstm':
            feature_size = hps.hidden_size * 2 
        else:
            feature_size = hps.num_filters_total
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)

        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            '''
            if hps.encode_method == 'cnn':
                article_feature = self._cnn_encoder(article)
                abstract_feature = self._cnn_encoder(abstract, is_article=False, reuse=True)
            '''
          
            with tf.variable_scope("article_emb"):
                article_W = tf.get_variable("article_W", [feature_size, hps.hidden_size],"float32", random_uniform_init)
                article_b = tf.get_variable("article_b", [hps.hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("abstract_emb"):
                abstract_W = tf.get_variable("abstract_W", [feature_size, hps.hidden_size],"float32", random_uniform_init)
                abstract_b = tf.get_variable("abstract_b", [hps.hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("scores_emb"):
                scores_W = tf.get_variable("scores_W", [hps.hidden_size, hps.num_classes], "float32", random_uniform_init)
                scores_b = tf.get_variable("scores_b", [hps.num_classes], "float32", random_uniform_init)

            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)
            with tf.variable_scope("output"):
                l2_loss += tf.nn.l2_loss(article_W)
                l2_loss += tf.nn.l2_loss(article_b)
                l2_loss += tf.nn.l2_loss(abstract_W)
                l2_loss += tf.nn.l2_loss(abstract_b)
                l2_loss += tf.nn.l2_loss(scores_W)
                l2_loss += tf.nn.l2_loss(scores_b)

                article_emb = tf.nn.xw_plus_b(article_feature, article_W, article_b)
                abstract_emb = tf.nn.xw_plus_b(abstract_feature, abstract_W, abstract_b)
                logits = tf.multiply(article_emb, abstract_emb)       # B,H
                score = tf.nn.xw_plus_b(logits, scores_W, scores_b)
                
                score_softmax = tf.nn.softmax(score)
                predictions = tf.argmax(score_softmax, 1, name="predictions") # (batch_size, ), type=int64
        return predictions, score_softmax, score, l2_loss

    def build_graph(self):
        hps = self._hps

        t0 = time.time()
        self._add_placeholders()

        '''Define loss'''
        # feature
        if hps.encode_method == 'cnn':
            art_feat = self._cnn_encoder(self._art_batch)
            ref_feat = self._cnn_encoder(self._ref_batch, is_article=False, reuse=True)
            ext_feat = self._cnn_encoder(self._ext_batch, is_article=False, reuse=True)
            wrong_feat = self._cnn_encoder(self._wrong_batch, is_article=False, reuse=True)
        elif hps.encode_method == 'lstm':
            art_feat = self._lstm_encoder(self._art_batch, self._art_lens)
            ref_feat = self._lstm_encoder(self._ref_batch, self._ref_lens, is_article=False, reuse=True)
            ext_feat = self._lstm_encoder(self._ext_batch, self._ext_lens, is_article=False, reuse=True)
            wrong_feat = self._lstm_encoder(self._wrong_batch, self._wrong_lens, is_article=False, reuse=True)

        # take the sample as fake data
        #self.fake_length = tf.reduce_sum(tf.stop_gradient(self.predict_mask),1)
        ref_pred, _, ref_logits, _ = self._text_classifier(art_feat, ref_feat)
        ext_pred, _, ext_logits, _ = self._text_classifier(art_feat, ext_feat, reuse=True)
        wrong_pred, _, wrong_logits, _ = self._text_classifier(art_feat, wrong_feat, reuse=True)

        # labels have shape [batch_size, num_classes], dtype = float32
        ref_labels = tf.concat((tf.ones((hps.batch_size,1)), tf.zeros((hps.batch_size,1)), tf.zeros((hps.batch_size,1))), 1)
        ext_labels = tf.concat((tf.zeros((hps.batch_size,1)), tf.ones((hps.batch_size,1)), tf.zeros((hps.batch_size,1))), 1)
        wrong_labels = tf.concat((tf.zeros((hps.batch_size,1)), tf.zeros((hps.batch_size,1)), tf.ones((hps.batch_size,1))), 1)

        ref_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ref_logits, labels=ref_labels))
        ext_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ext_logits, labels=ext_labels))
        wrong_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=wrong_logits, labels=wrong_labels))

        self.loss = ref_loss + ext_loss + wrong_loss
        tf.summary.scalar('loss', self.loss)

        self.ref_accu = tf.reduce_mean(tf.cast(tf.equal(ref_pred, tf.argmax(ref_labels, 1)), tf.float32))
        self.ext_accu = tf.reduce_mean(tf.cast(tf.equal(ext_pred, tf.argmax(ext_labels, 1)), tf.float32))
        self.wrong_accu = tf.reduce_mean(tf.cast(tf.equal(wrong_pred, tf.argmax(wrong_labels, 1)), tf.float32))
        self.total_accu = (self.ref_accu + self.ext_accu + self.wrong_accu) / 3.0
        tf.summary.scalar('ref_accu', self.ref_accu)
        tf.summary.scalar('ext_accu', self.ext_accu)
        tf.summary.scalar('wrong_accu', self.wrong_accu)
        tf.summary.scalar('total_accu', self.total_accu)

        '''Optimizer'''
        self.optim = tf.train.AdamOptimizer(hps.learning_rate)
        self.train_op = self.optim.minimize(self.loss)

        self.summaries = tf.summary.merge_all()
        t1 = time.time()
        print '[Info] seconds for building graph: ', t1-t0


    def _run_train_step(self, batch):
        hps = self._hps
        feed_dict = self._make_feed_dict(batch)

        to_return = {
            'train_op': self.train_op,
            'loss': self.loss,
            'ref_accu': self.ref_accu,
            'ext_accu': self.ext_accu,
            'wrong_accu': self.wrong_accu,
            'total_accu': self.total_accu,
            'summaries': self.summaries
        }

        return self._sess.run(to_return, feed_dict)

    def _run_eval_step(self, batch):
        hps = self._hps
        feed_dict = self._make_feed_dict(batch)

        to_return = {
            'loss': self.loss,
            'ref_accu': self.ref_accu,
            'ext_accu': self.ext_accu,
            'wrong_accu': self.wrong_accu,
            'total_accu': self.total_accu
        }

        return self._sess.run(to_return, feed_dict)

    def _eval_epoch(self, test_data):
        loss = []
        ref_accu = []
        ext_accu = []
        wrong_accu = []
        total_accu = []

        while 1:
            batch = test_data.next_batch()
            if not batch:
                break
            results = self._run_eval_step(batch)
            loss.append(results['loss'])
            ref_accu.append(results['ref_accu'])
            ext_accu.append(results['ext_accu'])
            wrong_accu.append(results['wrong_accu'])
            total_accu.append(results['total_accu'])

        avg_loss = sum(loss)/float(len(loss))
        avg_ref_accu = sum(ref_accu)/float(len(ref_accu))
        avg_ext_accu = sum(ext_accu)/float(len(ext_accu))
        avg_wrong_accu = sum(wrong_accu)/float(len(wrong_accu))
        avg_total_accu = sum(total_accu)/float(len(total_accu))
        return avg_loss, avg_ref_accu, avg_ext_accu, avg_wrong_accu, avg_total_accu

    def train(self):
        hps = self._hps
        train_data = self._train_data
        train_dir = os.path.join('.', hps.log_dir, hps.model_name, 'train')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        
        tf.initialize_all_variables().run()
        self.writer = tf.summary.FileWriter(train_dir, self._sess.graph)
        self.saver = tf.train.Saver(max_to_keep=hps.max_to_keep)
        
        # load cjeckpoint
        if hps.load_ckpt_path:
            self.saver.restore(self._sess, hps.load_ckpt_path)
            cur_iter = int(hps.load_ckpt_path.split('-')[-1])
            print '[Info] Load checkpoint: ', hps.load_ckpt_path
        else:
            try:
                ckpt_state = tf.train.get_checkpoint_state(train_dir)
                ckpt_path = ckpt_state.model_checkpoint_path
                self.saver.restore(self._sess, ckpt_path)
                cur_iter = int(ckpt_path.split('-')[-1])
                print '[Info] Load checkpoint: ', ckpt_path
            except:
                cur_iter = 1

        # train iteration
        for _ in range(hps.max_iter):
            batch = train_data.next_batch()
            t0 = time.time()
            results = self._run_train_step(batch)
            t1 = time.time()
            self.writer.add_summary(results['summaries'], cur_iter)
            print '[Info] Train step: %d\n loss: %f\n ref accu: %f\n ext accu: %f\n wro accu: %f\n avg accu: %f' % \
                  (cur_iter, results['loss'], results['ref_accu'], results['ext_accu'], \
                                             results['wrong_accu'], results['total_accu'])
            print ' seconds for training step: ', t1-t0, '\n'

            if cur_iter % 100 == 0:
                self.saver.save(self._sess, os.path.join(train_dir, 'model.ckpt'), global_step=cur_iter)
            if cur_iter % 100 == 0:
                t0 = time.time()
                test_loss, test_ref_accu, test_ext_accu, test_wrong_accu, test_total_accu = self._eval_epoch(self._test_data)
                t1 = time.time()
                print '---------------------------------------'
                print '[Info] Test Results: \n loss: %f\n ref accu: %f\n ext accu: %f\n wro accu: %f\n avg accu: %f' % \
                         (test_loss, test_ref_accu, test_ext_accu, test_wrong_accu, test_total_accu)
                print ' seconds for eval epoch: ', t1-t0, '\n'
                print '---------------------------------------'
            cur_iter += 1    

