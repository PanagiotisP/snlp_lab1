Files that need to have execution permission:

dict_alph_fst.py
input_fst.py
compile_fst.sh
spell_check.sh

aclImdb and GoogleNews-vectors-negative300.bin need to be in emotion directory

To create dictionary acceptors and Levenshtein FST:

./compile_fst.sh -p FILENAME

To correct word with word level model:

./spell_check.sh -w WORD

To correct word with character level model:

./spell_check.sh -l WORD