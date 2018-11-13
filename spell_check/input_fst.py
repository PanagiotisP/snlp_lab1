#!/usr/bin/python3

from create_fst import format_arc
import sys

# fst which accepts input word
def create_input (inputWord):
    f = open('input.fst.txt', 'w')
    i = 0
    for letter in inputWord:
        f.write(format_arc(i, i+1, letter, letter, 0))
        i += 1
    f.write(str(i) + '\n')
    f.close()

create_input(sys.argv[1])
