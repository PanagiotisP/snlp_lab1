# returns a string with the required format
def format_arc (src, dst, src_label, dst_label, weight):
    return str(src) + ' ' + str(dst) + ' ' + src_label + ' ' + dst_label + ' ' + str(weight) + '\n'

# fst with every single transitions
def create_levenshtein_fst (alphabet, delW = 1,  insW = 1, subW = 1):
    f = open('lev.fst.txt', 'w')

    for letter in alphabet:
        f.write(format_arc(0, 0, '<epsilon>', letter, insW))
        f.write(format_arc(0, 0, letter, '<epsilon>', delW))

        for otherletter in alphabet:
            f.write(format_arc(0, 0, letter, otherletter, 0 if letter == otherletter else subW))

    f.write('0\n')
    f.close()

# fst wich accepts every word of the dictionary, with 0 cost
def create_dictionary_fst (dictionary):
    f = open('dictionary.fst.txt', 'w')
    i = 0
    for word in dictionary:
        i += 1
        f.write(format_arc(0, i, word[0], word[0], 0))
        for letter in word[1:]:
            f.write(format_arc(i, i+1, letter, letter, 0))
            i += 1
        f.write(str(i) + '\n')
    f.close()
