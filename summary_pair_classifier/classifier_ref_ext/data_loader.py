import os
import json
import numpy as np
import data_utils
import random
from collections import defaultdict
import pdb

class CNNDMDataset():
    """CNN / Daily Mail summarization dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, hps, split, vocab):
        """Set the path for data(train, val or test) and vocabulary wrapper.

        Args:
            data_path: train, val or test pickle data path.
            vocab: vocabulary wrapper.
        """
        self.hps = hps
        self.split = split
        if self.split == 'train':
            self.data_dir = hps.train_dir
        elif self.split == 'test':
            self.data_dir = hps.test_dir

        self.data = defaultdict(list)
        for f in os.listdir(self.data_dir):
            data = json.load(open(os.path.join(self.data_dir, f)))
            self.data['article'].append(data['article'])
            self.data['ref'].append(data['ref'])
            self.data['ext'].append(data['ext'])
        
        self.num_data = len(self.data['article'])

        self.vocab = vocab
        self.vocab_size = self.vocab.size()

        self.current = 0
        if split == 'train':
            self.random_shuffle()

    def random_shuffle(self):
        paired_insts = list(zip(self.data['article'], self.data['ref'], self.data['ext']))
        random.shuffle(paired_insts)
        self.data['article'], self.data['ref'], self.data['ext'] = zip(*paired_insts)

    def next_batch(self):
        batch_size = self.hps.batch_size
        end = (self.current + batch_size) % self.num_data
        if self.current + batch_size < self.num_data:
            articles = self.data['article'][self.current:end]
            refs = self.data['ref'][self.current:end]
            exts = self.data['ext'][self.current:end]
        elif self.split == 'train':
            articles = self.data['article'][self.current:] + self.data['article'][:end]
            refs = self.data['ref'][self.current:] + self.data['ref'][:end]
            exts = self.data['ext'][self.current:] + self.data['ext'][:end]
            self.random_shuffle()
        else:
            self.current = 0
            return None

        #batch = {'article': [], 'ref': [], 'ext': [], 'art_len': [], 'ref_len': [], 'ext_len': []}
        batch = defaultdict(list)

        for i in range(batch_size):
            ids, length = data_utils.words2ids(articles[i].split(), \
                                               self.vocab, self.hps.max_article_len)
            batch['article'].append(ids)
            batch['art_len'].append(length)

            ids, length = data_utils.words2ids((' '.join(refs[i])).split(), \
                                               self.vocab, self.hps.max_abstract_len)
            batch['ref'].append(ids)
            batch['ref_len'].append(length)

            ids, length = data_utils.words2ids((' '.join(exts[i])).split(), \
                                                self.vocab, self.hps.max_abstract_len)
            batch['ext'].append(ids)
            batch['ext_len'].append(length)
        
        batch['article'] = np.array(batch['article'])
        batch['ref'] = np.array(batch['ref'])
        batch['ext'] = np.array(batch['ext'])
        self.current = end
        return batch


