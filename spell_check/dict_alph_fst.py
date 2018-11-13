#!/usr/bin/python3

import sys
from preprocess import create_probability_dictionary_alphabet, create_inout_symbols, preprocess_file, tokenize

corpus = sys.argv[1]
dictionary, alphabet = create_probability_dictionary_alphabet(preprocess_file(corpus, tokenize))
create_inout_symbols(alphabet)

from create_fst import create_levenshtein_fst, create_dictionary_fst
from statistics import mean
meanWordWeight = mean(dictionary.values())
meanLetterWeight = mean(alphabet.values())

create_levenshtein_fst(alphabet, meanWordWeight, meanWordWeight, meanWordWeight, 'word.fst')
create_levenshtein_fst(alphabet, meanLetterWeight, meanLetterWeight, meanLetterWeight, 'letter.fst')

create_dictionary_fst(dictionary, 'word.fst')

dictionary = {key:sum([alphabet[letter] for letter in key]) for key in dictionary}
create_dictionary_fst(dictionary, 'letter.fst')

# create dictionary as txt file
f = open('dict.txt', 'w')
for word in dictionary:
    f.write(word + ' \n')

f.close()
