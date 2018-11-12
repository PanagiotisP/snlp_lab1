usage() {
	echo "usage: ./spell_check.sh [-w || -l] WORD"
}

# check if there is an argument
if [ "$#" -ne 2 ]; then
	usage
	exit 0
fi

# parse options
if [ "$1" = "-w" ]; then
	DICT="dictionary.word.fst"
	LEV="lev.word.fst"
elif [ "$1" = "-l" ]; then
	DICT="dictionary.letter.fst"
	LEV="lev.letter.fst"
else
	usage
	exit 0
fi

# make an acceptor of the word
./input_fst.py "$2"

# compile word acceptor
fstcompile --isymbols=chars.syms --osymbols=chars.syms input.fst.txt input.fst

# compose input, levenshtain, dictionary fsts
fstcompose input.fst "$LEV" |
fstcompose - "$DICT" > input_lev_dic.fst


# find shortests paths
fstshortestpath input_lev_dic.fst |
fsttopsort |
fstarcsort > out.fst

# print output
fstprint --isymbols=chars.syms --osymbols=chars.syms out.fst |
cut -f4 |
grep -v "<epsilon>" |
head -n -1 |
tr -d '\n'
echo

# draw output fst
#fstdraw --isymbols=chars.syms --osymbols=chars.syms out.fst |
#dot -Tps > out.ps
