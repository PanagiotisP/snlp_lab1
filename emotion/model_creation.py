from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss
from gensim.models import Word2Vec, KeyedVectors
from preprocesslib import read_samples, preprocess, create_corpus, tokenize
import numpy as np

import os

data_dir = 'aclImdb'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
pos_train_dir = os.path.join(train_dir, 'pos')
neg_train_dir = os.path.join(train_dir, 'neg')
pos_test_dir = os.path.join(test_dir, 'pos')
neg_test_dir = os.path.join(test_dir, 'neg')

MAX_NUM_SAMPLES = 5000

NUM_W2V_TO_LOAD = 1000000

SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(42)

def create_model(features, y, clf = LogisticRegression()):
    clf.fit(features, y)
    return clf

def evaluate_model(model, features, y, metric = zero_one_loss):
    clf = model
    return metric(y, clf.predict(features))


# read positive and negative reviews
posTraining = read_samples(pos_train_dir, preprocess, MAX_NUM_SAMPLES)
print('Read posTraining')
negTraining = read_samples(neg_train_dir, preprocess, MAX_NUM_SAMPLES)
print('Read negTraining')
posTest = read_samples(pos_test_dir, preprocess, MAX_NUM_SAMPLES)
print('Read posTest')
negTest = read_samples(neg_test_dir, preprocess, MAX_NUM_SAMPLES)
print('Read negTest')

# create corpus
trainingCorpus, yTraining = create_corpus(posTraining, negTraining)
print('Made training corpus')
testCorpus, yTest = create_corpus(posTest, negTest)
print('Made test corpus')

# create vectorizer
vectorizer = CountVectorizer()
# fit on training corpus and extract feature to make training set
trainingSet = vectorizer.fit_transform(trainingCorpus).toarray()
# extract features from test corpus
testSet = vectorizer.transform(testCorpus).toarray()
# create model
model = create_model(trainingSet, yTraining)
# compute error on training and test sets
training_error = evaluate_model(model, trainingSet, yTraining)
test_error =  evaluate_model(model, testSet, yTest)

print('CountVectorizer:')
print('\ttraining error: {}'.format(training_error))
print('\ttest error: {}'.format(test_error))
print()

# a bit of memory management
del vectorizer, trainingSet, testSet, model

# same as before
vectorizer = TfidfVectorizer()
trainingSet = vectorizer.fit_transform(trainingCorpus).toarray()
testSet = vectorizer.transform(testCorpus).toarray()
model = create_model(trainingSet, yTraining)
training_error = evaluate_model(model, trainingSet, yTraining)
test_error =  evaluate_model(model, testSet, yTest)
print('TfidfVectorizer:')
print('\ttraining error: {}'.format(training_error))
print('\ttest error: {}'.format(test_error))
print()

# keep idf weights and vocabulary and 
# analyzer (preproxessor - tokenizer) for use with
# GoogleNews vectors
idf = vectorizer.idf_
idfIndex = vectorizer.vocabulary_
analyzer = vectorizer.build_analyzer()

del vectorizer, trainingSet, testSet, model


# load precomputed word2vec model
w2v = Word2Vec.load('W2VModel.model')

# function that returns vector represantation of comment
def comment2vec(comment):
    return np.mean([w2v.wv[word] if word in w2v.wv else np.zeros(w2v.vector_size) for word in tokenize(comment)], axis = 0)


vocab = set(word for comment in trainingCorpus for word in tokenize(comment))

# Out Of Vocabulary ratio
OOV = np.fromiter((0 if word in w2v.wv else 1 for word in vocab), dtype = 'int32').mean()
print()
print('Out of vocabulary words percentage for our word2vec: {}'.format(OOV))
print()

trainingSet = np.array([comment2vec(comment) for comment in trainingCorpus])
testSet = np.array([comment2vec(comment) for comment in testCorpus])
model = create_model(trainingSet, yTraining)
training_error = evaluate_model(model, trainingSet, yTraining)
test_error =  evaluate_model(model, testSet, yTest)
print('Our word2vec:')
print('\ttraining error: {}'.format(training_error))
print('\ttest error: {}'.format(test_error))
print()

del w2v, trainingSet, testSet, model


# load precomputed word2vec keyd vectors
w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True,  limit = NUM_W2V_TO_LOAD)


def comment2vec(comment):
    return np.mean([w2v[word] if word in w2v else np.zeros(w2v.vector_size) for word in tokenize(comment)], axis = 0)

OOV = np.fromiter((0 if word in w2v else 1 for word in vocab), dtype = 'int32').mean()
print()
print('Out of vocabulary words percentage for GoogleNes word2vec: {}'.format(OOV))
print()

trainingSet = np.array([comment2vec(comment) for comment in trainingCorpus])
testSet = np.array([comment2vec(comment) for comment in testCorpus])
model = create_model(trainingSet, yTraining)
training_error = evaluate_model(model, trainingSet, yTraining)
test_error =  evaluate_model(model, testSet, yTest)
print('GoogleNews word2vec:')
print('\ttraining error: {}'.format(training_error))
print('\ttest error: {}'.format(test_error))
print()


del trainingSet, testSet, model

# function that returns vector represantation of comment
# this time using tfidf weights
def comment2vec(comment):
    return np.mean([idf[idfIndex[word]] * w2v[word] if (word in w2v and word in idfIndex) else np.zeros(w2v.vector_size) for word in analyzer(comment)], axis = 0)

trainingSet = np.array([comment2vec(comment) for comment in trainingCorpus])
testSet = np.array([comment2vec(comment) for comment in testCorpus])
model = create_model(trainingSet, yTraining)
training_error = evaluate_model(model, trainingSet, yTraining)
test_error =  evaluate_model(model, testSet, yTest)
print('GoogleNews word2vec with tfidf:')
print('\ttraining error: {}'.format(training_error))
print('\ttest error: {}'.format(test_error))