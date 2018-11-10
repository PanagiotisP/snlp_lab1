#!/usr/bin/python3

import sys
from preprocess import create_dictionary_alphabet, create_inout_symbols, preprocess_file, tokenize

corpus = sys.argv[1]
dictionary, alphabet = create_dictionary_alphabet(preprocess_file(corpus, tokenize))
create_inout_symbols(alphabet)

from create_fst import create_levenshtein_fst, create_dictionary_fst

create_levenshtein_fst(alphabet)
create_dictionary_fst(dictionary)

# create dictionary as txt file
f = open('dict.txt', 'w')

for word in dictionary:
    f.write(word + ' \n')

f.close()
