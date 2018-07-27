# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 02:11:18 2017

@author: Changdong
"""

import pprint
from konlpy.tag import Twitter
from gensim.models.word2vec import Word2Vec
import re
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# tokenizing text using morpheme analyzer
pos_tagger = Twitter()
def pos(doc):
    return [t[0] for t in pos_tagger.pos(doc, norm=True, stem=True)
            if bool(re.search("Adjective|Noun|Adverb|Verb", t[1])) is True]

lib = pd.read_csv('moralpolitics/data/raw/민주2012.csv',
                  names=['url', 'date', 'article'],
                  engine='python', encoding='utf8').loc[1:]
cons = pd.read_csv('moralpolitics/data/raw/새누리2012.csv',
                   names=['url', 'date', 'article'],
                   engine='python', encoding='utf8')

# filter data by a length of articles
lib = lib[lib['article'].apply(lambda x: len(x) > 50)]
cons = cons[cons['article'].apply(lambda x: len(x) > 50)]

lib['article'] = [pos(i) for i in lib['article']]
cons['article'] = [pos(i) for i in cons['article']]


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

# frequency

from numpy import std, mean, sqrt
from scipy import stats
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)

for mf in MFDictionary:
    lcount = []
    for i in lib:
        lcount.append(len([j for j in i if j in eval(mf)])/len(i))
    ccount = []
    for i in cons:
        ccount.append(len([j for j in i if j in eval(mf)])/len(i))

    print(mf, mean(ccount)*100, mean(lcount)*100, 
          stats.ttest_ind(ccount, lcount)[0], stats.ttest_ind(ccount, lcount)[1],
          cohen_d(ccount, lcount))


# w2v model
model = Word2Vec(cons['article'], size=300, alpha=0.025, min_alpha=0.025,
                 window=10, workers=14)
model.build_vocab(cons['article'])
model.save('moralpolitics/data/2012.word2vec') #vector space 저장
model = Word2Vec.load('moralpolitics/data/2012.word2vec')

# load the mfd (only includes words in the w2v model)
MFDictionary=['HarmVirtue','HarmVice','FairnessVirtue','FairnessVice',
              'IngroupVirtue','IngroupVice','AuthorityVirtue','AuthorityVice',
              'PurityVirtue','PurityVice']
for i in MFDictionary:
    wordlist = []
    for j in read_mfd('moralpolitics/data/mfd/{0}.csv'.format(i)):
        if pos(j) == []:
            next
        else:
            print(j)
            print(pos(j))
            Morp = pos(j)[0]
            wordlist.append(Morp)
            wordlist = [i for i in wordlist if i in model.wv.index2word]
            exec("%s=%s" % (i, unique_list(wordlist)))

libcare = model.most_similar(FairnessVirtue, FairnessVice, topn = 50)
libfair = model.most_similar(HarmVirtue, HarmVice, topn = 50)
libing = model.most_similar(IngroupVirtue, IngroupVice, topn = 50)
libaut = model.most_similar(AuthorityVirtue, AuthorityVice, topn = 50)
libpur = model.most_similar(PurityVirtue, PurityVirtue, topn = 50)


# frequency

for mf in MFDictionary:
    lcount = []
    for i in lib:
        lcount.append(len([j for j in i if j in mf]))
    ccount = []
    for i in cons:
        ccount.append(len([j for j in i if j in mf]))
        
    print(mf, np.mean(lcount), np.mean(ccount))
        
    libharmvirtue = [[j] for i in lib for j in i if j in mf]
    conharmvirtue = [[j] for i in cons for j in i if j in mf]


model.most_similar

import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
# matplotlib의 글자가 꺠지지 않게 하기 위해 matplotlib의 폰트를 바꾸어줌
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\malgun.ttf").get_name()
rc('font', family=font_name)
flist = matplotlib.font_manager.get_fontconfig_fonts()

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='pca')
from sklearn.cluster import KMeans
import pandas as pd

names = [j for i in MFDictionary for j in eval(i)]
mfvectors = [model[i] for i in names]

result = pd.DataFrame({'names': names, 'vectors': mfvectors})

colors = np.hstack([np.repeat(['#FFFF00'], len(HarmVirtue) + len(HarmVice)),
                   np.repeat(['#FF3300'], len(FairnessVirtue) + len(FairnessVice)),
                   np.repeat(['#00CC33'], len(IngroupVirtue) + len(IngroupVice)),
                   np.repeat(['#6699FF'], len(AuthorityVirtue) + len(AuthorityVice)),
                   np.repeat(['#9900CC'], len(PurityVirtue) + len(PurityVice))])


textplot = tsne.fit_transform(mfvectors)

xlist = []
ylist = []
for x, y  in textplot:
    xlist.append(x)
    ylist.append(y)
plt.figure(figsize=(30,30))
plt.scatter(xlist, ylist, c=colors)
rc('font',family=font_name, size=10)
for k, v, s in zip(xlist, ylist, names):
    plt.text(k,v,s)
    
plt.savefig('moralpolitics/test.png')

#전체적으로 리뉴얼 필요

# poll

support = pd.read_csv('moralpolitics/data/poll.txt',
                      names=['date', 'lib', 'cons', 'source', 'pubdate'],
                      engine='python', encoding='utf8')

support['date'] = [datetime.strptime(i, '%Y-%m-%d') for i in support['date']]

support = support.sort_values(by=['date'])
