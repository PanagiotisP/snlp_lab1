
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import zero_one_loss
import numpy as np
from preprocesslib import *

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
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000

import numpy as np

SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(42)

def create_model(features, y):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    clf = LogisticRegression()
    clf.fit(features, y)
    return scaler, clf

def evaluate_model(model, features, y, metric = zero_one_loss):
    scaler, clf = model
    features = scaler.transform(features)
    return metric(y, clf.predict(features))


posTraining = read_samples(pos_train_dir, preprocess)
negTraining = read_samples(neg_train_dir, preprocess)
posTest = read_samples(neg_test_dir, preprocess)
negTest = read_samples(neg_test_dir, preprocess)

trainingCorpus, yTraining = create_corpus(posTraining, negTraining)
testCorpus, yTest = create_corpus(posTest, negTest)

CBOWvectorizer = CountVectorizer()
CBOWTraining = CBOWvectorizer.fit_transform(trainingCorpus)
CBOWModel = create_model(CBOWTraining, yTraining)
print(evaluate_model(CBOWModel, CBOWTraining, yTraining))
#CBOWTest = CBOWvectorizer.fit_transform(testCorpus)
