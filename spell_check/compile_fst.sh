#!/usr/bin/env bash

usage() {
	echo "usage: ./compile_fst.sh [-p FILENAME]"
}

# parse options
while getopts hp: opt; do
	case $opt in
		h)
			usage
			exit 0
			;;
		p)
			if [ -e ${OPTARG} ]; then
				./dict_alph_fst.py ${OPTARG}
			else
				echo "${OPTARG} does not exist"
				exit 1
			fi
			;;
		*)
			usage
			exit 1
			;;
	esac
done

# compile dictionary acceptors

fstcompile --isymbols=chars.syms --osymbols=chars.syms dictionary.word.fst.txt |
# fstrmepsilon not needed
fstdeterminize |
fstminimize - |
fstarcsort --sort_type=ilabel - dictionary.word.fst

fstcompile --isymbols=chars.syms --osymbols=chars.syms dictionary.letter.fst.txt |
# fstrmepsilon not needed
fstdeterminize |
fstminimize - |
fstarcsort --sort_type=ilabel - dictionary.letter.fst

# compile levenshtein transducers

fstcompile --isymbols=chars.syms --osymbols=chars.syms lev.word.fst.txt |
fstarcsort --sort_type=ilabel - lev.word.fst

fstcompile --isymbols=chars.syms --osymbols=chars.syms lev.word.fst.txt |
fstarcsort --sort_type=ilabel - lev.letter.fst
