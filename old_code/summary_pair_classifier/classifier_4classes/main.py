import os
import time
import numpy as np
from data_utils import Vocab
from data_loader import CNNDMDataset
from model import TextClassifier
import tensorflow as tf
import pprint
import pdb

flags = tf.app.flags
flags.DEFINE_string("train_dir", 'data/CNNDM/train', "")
flags.DEFINE_string("test_dir", 'data/CNNDM/test', "")
flags.DEFINE_string("vocab_path", 'data/CNNDM/vocab', "")
flags.DEFINE_integer("vocab_size", 50000, "")

flags.DEFINE_string("encode_method", 'cnn', "cnn / lstm")
flags.DEFINE_integer("num_classes", 4, "")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate of for adam [0.0003]")
flags.DEFINE_integer("hidden_size", 512, "")
flags.DEFINE_integer("max_article_len", 400, "")
flags.DEFINE_integer("max_abstract_len", 120, "")
flags.DEFINE_float("l2_reg_lambda", 0.2, "")
flags.DEFINE_float("drop_out_rate", 0.3, "")

flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")
#flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_integer("max_iter", 50000, "")
flags.DEFINE_integer('max_to_keep', 10, '')
flags.DEFINE_string("model_name", "3classes_ref_ext_wrong", "")
flags.DEFINE_string("log_dir", "log", "Directory name to save the checkpoints [checkpoint]")

flags.DEFINE_string("load_ckpt_path", None, "Path to the checkpoint you want to load")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1/10
    config.gpu_options.allow_growth = True

    t0 = time.time()
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)
    train_data = CNNDMDataset(FLAGS, 'train', vocab)
    test_data = CNNDMDataset(FLAGS, 'test', vocab)
    t1 = time.time()
    print '[Info] seconds for loading data: ', t1-t0

    with tf.Session(config=config) as sess:
        FLAGS.filter_sizes = [3,5,12,24,32]
        FLAGS.num_filters = [100,100,100,100,100]
        #FLAGS.filter_sizes = [1,2,3,4,5,6,7,8,9,10,16,24,32]
        #FLAGS.num_filters = [100,200,200,200,200,100,100,100,100,100,160,160,160]
        FLAGS.num_filters_total = sum(FLAGS.num_filters)

        if FLAGS.is_train:
            model = TextClassifier(sess, train_data, test_data, FLAGS)
            model.build_graph()
            model.train()
        else:
            model = TextClassifier(sess, train_data, test_data, FLAGS)
            model.build_graph()
            model.evaluate()

if __name__ == '__main__':
    tf.app.run()
