from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import zero_one_loss
from gensim.models import Word2Vec, KeyedVectors
from preprocesslib import read_samples, preprocess, create_corpus, tokenize
from statistics import mean
import numpy as np

import os

data_dir = '.\\aclImdb\\'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
pos_train_dir = os.path.join(train_dir, 'pos')
neg_train_dir = os.path.join(train_dir, 'neg')
pos_test_dir = os.path.join(test_dir, 'pos')
neg_test_dir = os.path.join(test_dir, 'neg')

MAX_NUM_SAMPLES = 1500

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


posTraining = read_samples(pos_train_dir, preprocess, MAX_NUM_SAMPLES)
negTraining = read_samples(neg_train_dir, preprocess, MAX_NUM_SAMPLES)
posTest = read_samples(pos_test_dir, preprocess, MAX_NUM_SAMPLES)
negTest = read_samples(neg_test_dir, preprocess, MAX_NUM_SAMPLES)

trainingCorpus, yTraining = create_corpus(posTraining, negTraining)
testCorpus, yTest = create_corpus(posTest, negTest)

BOWVectorizer = CountVectorizer()
BOWTraining = BOWVectorizer.fit_transform(trainingCorpus).toarray()
BOWTest = BOWVectorizer.transform(testCorpus).toarray()
BOWModel = create_model(BOWTraining, yTraining)
training_error = evaluate_model(BOWModel, BOWTraining, yTraining)
test_error =  evaluate_model(BOWModel, BOWTest, yTest)
print('Bag of words:')
print('\ttraining error: {}'.format(training_error))
print('\ttest error: {}'.format(test_error))

del BOWVectorizer, BOWTraining, BOWTest, BOWModel


WBOWVectorizer = TfidfVectorizer()
WBOWTraining = WBOWVectorizer.fit_transform(trainingCorpus).toarray()
WBOWTest = WBOWVectorizer.transform(testCorpus).toarray()
WBOWModel = create_model(WBOWTraining, yTraining)
training_error = evaluate_model(WBOWModel, WBOWTraining, yTraining)
test_error =  evaluate_model(WBOWModel, WBOWTest, yTest)
print('Weighted bag of words:')
print('\ttraining error: {}'.format(training_error))
print('\ttest error: {}'.format(test_error))

idf = WBOWVectorizer.idf_
idfIndex = WBOWVectorizer.vocabulary_
analyzer = WBOWVectorizer.build_analyzer()

del WBOWVectorizer, WBOWTraining, WBOWTest, WBOWModel


w2v = Word2Vec.load('W2VModel.model')

def comment2vec(comment):
    return [w2v.wv[word] if word in w2v.wv else np.zeros(w2v.vector_size) for word in tokenize(comment)]

vocab = set()
for comment in trainingCorpus:
    for word in tokenize(comment):
        vocab.add(word)

OOV = mean((0 if word in w2v.wv else 1 for word in vocab))
print()
print('Out of vocabulary words percentage for word2vec: {}'.format(OOV))

W2VTraining = [np.mean(comment2vec(comment), axis = 0) for comment in trainingCorpus]
W2VTest = [np.mean(comment2vec(comment), axis = 0) for comment in testCorpus]
W2VModel = create_model(W2VTraining, yTraining)
training_error = evaluate_model(W2VModel, W2VTraining, yTraining)
test_error =  evaluate_model(W2VModel, W2VTest, yTest)
print('Neural bag of words:')
print('\ttraining error: {}'.format(training_error))
print('\ttest error: {}'.format(test_error))

del w2v, W2VTraining, W2VTest, W2VModel


NUM_W2V_TO_LOAD = 500000

w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True,  limit = NUM_W2V_TO_LOAD)

def comment2vec(comment):
    return [w2v[word] if word in w2v else np.zeros(w2v.vector_size) for word in tokenize(comment)]

OOV = mean((0 if word in w2v else 1 for word in vocab))
print()
print('Out of vocabulary words percentage for word2vec: {}'.format(OOV))

W2VTraining = [np.mean(comment2vec(comment), axis = 0) for comment in trainingCorpus]
W2VTest = [np.mean(comment2vec(comment), axis = 0) for comment in testCorpus]
W2VModel = create_model(W2VTraining, yTraining)
training_error = evaluate_model(W2VModel, W2VTraining, yTraining)
test_error =  evaluate_model(W2VModel, W2VTest, yTest)
print('Neural bag of words:')
print('\ttraining error: {}'.format(training_error))
print('\ttest error: {}'.format(test_error))


del W2VTraining, W2VTest, W2VModel

def comment2vec(comment):
    return [idf[idfIndex[word]] * w2v[word] if (word in w2v and word in idfIndex) else np.zeros(w2v.vector_size) for word in analyzer(comment)]

W2VTraining = [np.mean(comment2vec(comment), axis = 0) for comment in trainingCorpus]
W2VTest = [np.mean(comment2vec(comment), axis = 0) for comment in testCorpus]
W2VModel = create_model(W2VTraining, yTraining)
training_error = evaluate_model(W2VModel, W2VTraining, yTraining)
test_error =  evaluate_model(W2VModel, W2VTest, yTest)
print('Neural bag of words:')
print('\ttraining error: {}'.format(training_error))
print('\ttest error: {}'.format(test_error))