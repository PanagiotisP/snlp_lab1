def identity_preprocess(stringToPreprocess):
    return stringToPreprocess


def preprocess_file(filePath, preprocessFun = identity_preprocess):
    f = open(filePath, 'r')

    # initialize list
    preprocessedList = preprocessFun(f.readline())
    for line in f:
        preprocessedList += preprocessFun(line)
    f.close()
    return preprocessedList

def produce_sentences(filePath):
    with open(filePath, 'r') as f:
        return [tokenize(sentence) for sentence in f.read().split('.')]

import re

# replaces any non alphabetic character with a <space>
def strip_punctuation(line):
     return re.sub(r'[^a-zA-Z\s]', r' ', line)

# replaces multiple whitespaces with a single one and produces tokenized list
def tokenize(line):
    line = line.strip()
    return [word for word in re.sub(r'\s+', r' ', strip_punctuation(line)).lower().split(' ') if word != '']

# creates dictionary and alphabet as sets
def create_dictionary_alphabet(tokenizedList):
    dictionary = set(tokenizedList)
    alphabet = set(''.join(tokenizedList))
    return dictionary, alphabet

# creates input and output symbols for fst
def create_inout_symbols(alphabet):
    f = open('chars.syms', 'w')
    f.write('<epsilon>\t0\n')

    for index, character in enumerate(alphabet, 1):
        f.write(character + '\t' + str(index) + '\n')
    f.close()
