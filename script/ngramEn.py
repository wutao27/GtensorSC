import os
import math
import itertools
CUR_DIR = os.getcwd()
DATA_DIR = CUR_DIR + "/../raw_data/"
FILE_NAME = ["w3_.txt", "w4_.txt"]
TENSOR_DIM = [3,4]

def get_word_dic(textFile, n):
    wordDic = {}; index = 1
    f = open(textFile,'r')
    oneline = f.readline()
    while (oneline!=""):
        entry = oneline.split()
        for i in range(n):
            if entry[i+1] not in wordDic:
                wordDic[entry[i+1]] = index
                index += 1
        oneline = f.readline()
    return wordDic


def get_tensor_dic(textFile, wordDic, n):
    tensorDic = {}
    f = open(textFile,'r')
    oneline = f.readline()
    while(oneline!=""):
        tempList = []
        entry = oneline.split()
        for i in range(n):
            tempList.append(wordDic[ entry[i+1] ])
        tensorDic[tuple(tempList)] = int(entry[0])
        oneline = f.readline()
    return tensorDic


def get_data_dic(tensorDic, n):
    dataDic = {}
    
    flagDic = {}
    for item in tensorDic:
        flagDic[item] = False
        
    for item in tensorDic:
        if flagDic[item] is True:
            continue
        flagDic[item] = True
        ip = itertools.permutations(item)
        tSum = 0
        for ipItem in ip:
            if ipItem in tensorDic:
                tSum += tensorDic[ipItem]
                flagDic[ipItem] = True
            
        ip = itertools.permutations(item)
        for ipItem in ip:
            dataDic[ipItem] = tSum
    
    return dataDic


def print_file(textFile, dicFile, dataDic, wordDic, n):
    
    f = open(textFile,'w')
    for item in dataDic:
        tempList = []
        for num in item:
            tempList.append(str(num))
        indStr = '  '.join(tempList)
        f.write(indStr + "  " + str(dataDic[item]) + "\n")
    f.close()
    
    f = open(dicFile,'w')
    for item in wordDic:
        f.write(str(wordDic[item]) + "  " + item + "\n")
    f.close()


def process_data(i):
    textFile = DATA_DIR + FILE_NAME[i]
    wordDic = get_word_dic(textFile,TENSOR_DIM[i])
    tensorDic = get_tensor_dic(textFile, wordDic, TENSOR_DIM[i])
    dataDic = get_data_dic(tensorDic, TENSOR_DIM[i])
    print_file(CUR_DIR+"/../data/"+"tensor_"+FILE_NAME[i], CUR_DIR+"/../data/"+"dic_"+FILE_NAME[i], dataDic, wordDic, TENSOR_DIM[i])
    
    
process_data(0)
process_data(1)