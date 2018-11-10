usage() {
	echo "usage: ./spell_check.sh WORD"
}

# check if there is an argument
if [ -z "$1" ]; then
	usage
	exit 0
fi

# make an acceptor of the word
./input_fst.py "$1"

# compile word acceptor
fstcompile --isymbols=chars.syms --osymbols=chars.syms input.fst.txt input.fst

# compose input, levenshtain, dictionary fsts
fstcompose input.fst lev.fst |
fstcompose - dictionary.fst > input_lev_dic.fst

# find shortests paths
fstshortestpath --nshortest=5 input_lev_dic.fst |
fsttopsort |
fstarcsort > out.fst

# print output
fstprint --isymbols=chars.syms --osymbols=chars.syms out.fst |
# keep 4th (output characters) and 5th (weights) columns
cut --fields=4,5 |
# delete tabs
tr -d '\t' |
# delete last line
head -n -1 |
# if the following pattern occurs a deletion has happened 
sed -E 's/<epsilon>1//' |
tr -d '\n' |
# since we have taken care of deletions an <epsilon> indicates the end of a word
sed -E 's/(<epsilon>)+/\n/g' |
tr -d '1'
echo

# draw output fst
#fstdraw --isymbols=chars.syms --osymbols=chars.syms out.fst |
#dot -Tps > out.ps
