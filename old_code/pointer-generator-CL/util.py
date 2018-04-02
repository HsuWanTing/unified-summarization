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
FLAGS = tf.app.flags.FLAGS

def get_config():
  """Returns config for tf.session"""
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  return config

def load_ckpt(saver, sess, ckpt_path=None, wait_for_ckpt=False):
  """Load checkpoint from the train directory and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
  while True:
    train_dir = os.path.join(FLAGS.log_root, "train")
    try:
      if not ckpt_path:
        ckpt_state = tf.train.get_checkpoint_state(train_dir)
        ckpt_path = ckpt_state.model_checkpoint_path
      tf.logging.info('Loading checkpoint %s', ckpt_path)
      saver.restore(sess, ckpt_path)
      return ckpt_path
    except:
      if ckpt_path:
        sys.exit("Failed to load checkpoint from %s.", ckpt_path)
      if not wait_for_ckpt:
        tf.logging.info("Failed to load checkpoint from %s.", train_dir)
        break
      tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", train_dir, 10)
      time.sleep(10)
