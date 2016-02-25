import os
import math
import itertools

CUR_DIR = os.getcwd()
DATA_DIR = CUR_DIR + "/../raw_data/"

def get_word_dic(textFile, n):
    wordDic = {}; index = 1
    f = open(textFile,'r')
    oneline = f.readline()
    while (oneline!=""):
        uitem = unicode(oneline,"utf-8")
        uwords = uitem.split()
        for i in range(n):
            if uwords[i] not in wordDic:
                wordDic[uwords[i]] = index
                index += 1
        oneline = f.readline()
    return wordDic

def get_tensor_dic(textFile, wordDic, n):
    tensorDic = {}
    f = open(textFile,'r')
    oneline = f.readline()
    while(oneline!=""):
        tempList = []
        uitem = unicode(oneline,"utf-8")
        uwords = uitem.split()
        for i in range(n):
            tempList.append(wordDic[ uwords[i] ])
        tensorDic[tuple(tempList)] = int(uwords[n])
        oneline = f.readline()
    return tensorDic


def get_data_dic(tensorDic):
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
        f.write((item + "  " + str(wordDic[item]) + "\n").encode('utf8'))
    f.close()




textFile = DATA_DIR+"Chinese_3gram.csv";dim = 3
wordDic = get_word_dic(textFile,dim)
tensorDic = get_tensor_dic(textFile, wordDic, dim)
dataDic = get_data_dic(tensorDic)
print_file(CUR_DIR+"/../data/tensor_Chinese_3gram.txt", CUR_DIR+"/../data/dic_Chinese_3gram.txt", dataDic, wordDic, dim)

textFile = DATA_DIR+"Chinese_4gram.csv";dim = 4
wordDic = get_word_dic(textFile,dim)
tensorDic = get_tensor_dic(textFile, wordDic, dim)
dataDic = get_data_dic(tensorDic)
print_file(CUR_DIR+"/../data/tensor_Chinese_4gram.txt", CUR_DIR+"/../data/dic_Chinese_4gram.txt", dataDic, wordDic, dim)

