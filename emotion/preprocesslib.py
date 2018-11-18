import os

data_dir = './aclImdb/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
pos_train_dir = os.path.join(train_dir, 'pos')
neg_train_dir = os.path.join(train_dir, 'neg')
pos_test_dir = os.path.join(test_dir, 'pos')
neg_test_dir = os.path.join(test_dir, 'neg')

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 5000

import numpy as np

SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(42)

try:
    import glob2 as glob
except ImportError:
    import glob

import re

def strip_punctuation(s):
    return re.sub(r'[^a-zA-Z\s]', ' ', s)

def preprocess(s):
    return re.sub('\s+',' ', strip_punctuation(s).lower())

def tokenize(s):
    return s.split(' ')

def preproc_tok(s):
    return tokenize(preprocess(s))

def read_samples(folder, preprocess=lambda x: x, maxSamples = MAX_NUM_SAMPLES):
    samples = glob.iglob(os.path.join(folder, '*.txt'))
    data = []
    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == maxSamples:
            break
        with open(sample, mode = 'r', encoding = 'utf8') as fd:
            x = [preprocess(l) for l in fd][0]
            data.append(x)
    return data

def create_corpus(pos, neg):
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    return corpus[indices], y[indices]