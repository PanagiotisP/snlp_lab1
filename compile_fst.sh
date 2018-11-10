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

# compile dictionary acceptor
fstcompile --isymbols=chars.syms --osymbols=chars.syms dictionary.fst.txt |
# fstrmepsilon not needed
fstdeterminize |
fstminimize - dictionary.fst

# compile levenshtein transducer
fstcompile --isymbols=chars.syms --osymbols=chars.syms lev.fst.txt lev.fst
