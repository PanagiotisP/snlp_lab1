#!/usr/bin/python3

import os

fin = open("spell_checker_test_set.txt", 'r')

# output to both STDIn and a file
fout = open("evaluator_out.txt", 'w')

correctCounter = 0
testSize = 0

for line in fin:
    targetWord, testList = line.split(':')
    print(targetWord + ':')
    fout.write(targetWord + ':')

    testList = testList.strip().split(' ')
    for word in testList:
        testSize += 1

        # ./spell_check.sh gives several corrections
        corrections = os.popen('./spell_check.sh' + ' ' + word).read().strip().split('\n')
        print('\t' + word + ' -> ' + str(corrections))
        fout.write('\t' + word + ' -> ' + str(corrections))

        # if any correction matches the expected word increase counter
        for correction in corrections:
            if correction == targetWord:
                correctCounter += 1
                break
    print()
    fout.write('\n')

print(str(correctCounter) + ' correct out of ' + str(testSize) + ', ' + str(correctCounter / testSize * 100) + '%')
fout.write(str(correctCounter) + ' correct out of ' + str(testSize) + ', ' + str(correctCounter / testSize * 100) + '%\n')

fout.close()
fin.close()
