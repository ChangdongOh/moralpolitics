9# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 00:15:15 2017

@author: Changdong
"""

import re
import pickle 
import pandas as pd
import numpy as np
import csv
from konlpy.tag import Twitter 
from collections import Counter
from scipy import stats
from gensim.models.doc2vec import TaggedDocument



pos_tagger = Twitter()


def pos(doc):
    return [t[0] for t in pos_tagger.pos(doc, norm=True, stem=True)
            if bool(re.search("Adjective|Noun|Adverb|Verb", t[1])) is True]


def read_mfd(filename):
        with open(filename, 'r') as f:
            data = [line for line in f.read().splitlines()]

        return data


def unique_list(l):
    x = []
    for a in l:
        if a not in x:
            x.append(a)
    return x

rep = [["[가-힣]{2,4} 대변인|([가-힣]\s{1,3}){3}대변인|굴림", " "],
        ["진보당|통합진보당", "통진당"],
        ["진보정의당", "정의당"],
        [("통합민주당|통민당|민주캠프|문재인캠프|시민캠프|"
          "통합 민주당|열린우리당|대통합민주신당|통합신당|"
          "민주통합당|민통당|열린 우리당|새정치민주연합|더민주|"
          "더불어민주당|더불어 민주당|새민연|새민련|야당"), ' 민주당 '],
        ["한나라당|여당|한당", " 새누리당 "]]


def preprocess(t, rules=rep):

    for i in rules:
        t = [re.sub(i[0], i[1], j) for j in t]

    return t


def Tagged(year, party):
    print(party, str(year))
    tagged_docs = []

    def read_data(filename):
        with open(filename, 'r', encoding='UTF8') as f:
            data = pd.DataFrame([line.split(',')[1:3] for line in f.read().splitlines()],
                                 columns=['date', 'article']).drop_duplicates().reset_index(drop=True)
        
        data['article'] = preprocess(data['article'])

        return data
        
    data = read_data('moralpolitics/data/raw/'+party+'{0}.csv'.format(str(year)))

    if party == '새누리':
        data['article'] = preprocess(data['article'], rules=[["자당", "새누리당"],
                                     ["우리당|우리 당|저희 당|저희당", "민주당"]])
    else:
        data['article'] = preprocess(data['article'], rules=[["자당", "민주당"],
                                     ["우리당|우리 당|저희 당|저희당", "새누리당"]])

    newrule = [["따르면|드린다|들이|다음과 같이|굴림", " "]]
    data['article'] = [preprocess(pos(i), newrule) for i in data['article']]

    for i in range(0, len(data)):
        doc = data.ix[i]
        tagged_docs.append(TaggedDocument(doc[1],
                                          [party + doc[0][:-3] + ': ' + str(i)]))
        
    
    with open('moralpolitics/data/pos/' + party + str(year) + '.csv',
              'w') as f:
        wr = csv.writer(f, delimiter='\n')
        wr.writerow([' '.join(i) for i in list(data['article'])])

    return tagged_docs


total_docs = []
for year in range(2008, 2017):
    total_docs += Tagged(year, '새누리')+Tagged(year, '민주')

with open('moralpolitics/data/taggeddoc.txt', 'wb') as p:
    pickle.dump(total_docs, p)
'''
with open('moralpolitics/data/taggeddoc.txt', 'rb') as p:
    total_docs = pickle.load(p)
'''



for i in MFDictionary:
    wordlist = []
    for j in read_mfd('moralpolitics/data/mfd/{0}.csv'.format(i)):
        print(pos(j))
        Morp = pos(j)[0]
        wordlist.append(Morp)
    exec("%s=%s" % (i, unique_list(wordlist))) 

# frequency change of moral words

def mffreq(total_docs, party):
    y = [i for i in range(2008, 2017)]
    m = ['01', '02', '03', '04', '05', '06',
             '07', '08', '09', '10', '11', '12']
    totalfreq = {}
    
    for year in y:
        for month in m:
            print(str(year) + '-' + month)
            
            docs = [i[0] for i in total_docs
                    if party+str(year)+'-'+str(month) in i[1][0]]
            freq = {0:[], 1:[], 2:[], 3:[], 4:[]}
            for doc in docs:
                words = [i for i in doc]
                count = Counter(words)
                for mf in range(0, 5):
                    wordlist = [i for i in (eval(MFDictionary[2*mf]) +
                                            eval(MFDictionary[2*mf+1])) if i in words]
                    freq[mf].append(sum([count[word]/len(words) for word in wordlist]))
            meanse = [[np.mean(freq[mf]), stats.sem(freq[mf])] for mf in range(0, 5)]
            totalfreq[str(year) + '-' + month] = [j for i in meanse for j in i]
    
    totalfreq = pd.DataFrame(totalfreq, index=['Caremean', 'Carese',
                                               'Fairnessmean', 'Fairnessse',
                                               'Ingroupmean', 'Ingroupse',
                                               'Authoritymean', 'Authorityse',
                                               'Puritymean', 'Purityse'])
    totalfreq.to_csv('moralpolitics/result/freq/' + party + '.csv')

mffreq(total_docs, '민주')
mffreq(total_docs, '새누리')
