#!/usr/bin/python3

import os

fin = open("spell_checker_test_set.txt", 'r')

# output to both STDIn and a file
fout = open("evaluator_out.txt", 'w')

word_level_correctCounter = 0
letter_level_correctCounter = 0
testSize = 0

for line in fin:
    targetWord, testList = line.split(':')
    print(targetWord + ':')
    fout.write(targetWord + ':')

    testList = testList.strip().split(' ')
    for word in testList:
        testSize += 1

        # ./spell_check.sh gives several corrections
        word_level_correction = os.popen('./spell_check.sh' + ' -w ' + word).read().strip()
        letter_level_correction = os.popen('./spell_check.sh' + ' -l ' + word).read().strip()
        print('\t-w' + word + ' -> ' + word_level_correction)
        fout.write('\t-w' + word + ' -> ' + word_level_correction)
        print('\t-l' + word + ' -> ' + letter_level_correction)
        fout.write('\t-l' + word + ' -> ' + letter_level_correction)

        # if any correction matches the expected word increase counter
        if targetWord == word_level_correction:
            word_level_correctCounter += 1
        if targetWord == letter_level_correction:
            letter_level_correctCounter += 1
            
    print()
    fout.write('\n')

print(str(word_level_correctCounter) + ' correct out of ' + str(testSize) + ', ' + str(word_level_correctCounter / testSize * 100) + '%')
fout.write(str(word_level_correctCounter) + ' correct out of ' + str(testSize) + ', ' + str(word_level_correctCounter / testSize * 100) + '%')
print(str(letter_level_correctCounter) + ' correct out of ' + str(testSize) + ', ' + str(letter_level_correctCounter / testSize * 100) + '%')
fout.write(str(letter_level_correctCounter) + ' correct out of ' + str(testSize) + ', ' + str(letter_level_correctCounter / testSize * 100) + '%')

fout.close()
fin.close()