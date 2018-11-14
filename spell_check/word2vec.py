#!/usr/bin/python3

import numpy as np
from gensim.models import Word2Vec
from preprocess import produce_sentences
import sys

sentences = produce_sentences(sys.argv[1])

# Initialize word2vec. Context is taken as the 2 previous and 2 next words
model = Word2Vec(sentences, window=5, size=100, workers=10)
model.train(sentences, total_examples=len(sentences), epochs=100)
#f = open('W2VModel.model', 'w')
#model.save('W2VModel.model')
#f.close()

# get ordered vocabulary list
voc = model.wv.index2word

# get vector size
dim = model.vector_size

# get most similar words
for word in ['holmes', 'quick', 'murder', 'pipe', 'inspector', 'sir', 'wall', 'happy', 'red', 'swiftly']:
    sim = model.wv.most_similar(word)
    print(word + ':')
    for simWord in sim:
        print('\t',simWord)

# Convert to numpy 2d array (n_vocab x vector_size)
def to_embeddings_Matrix(model):
    embedding_matrix = np.zeros((len(model.wv.vocab), model.vector_size))
    word2idx = {}
    for i in range(len(model.wv.vocab)):
        embedding_matrix[i] = model.wv[model.wv.index2word[i]]
        word2idx[model.wv.index2word[i]] = i
    return embedding_matrix, model.wv.index2word, word2idx