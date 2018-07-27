# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 11:16:56 2017

@author: Changdong
"""


from konlpy.tag import Twitter
from gensim.models import doc2vec
import re
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

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

def meanvector(model, positive, negative):
    
    from six import string_types, integer_types, itervalues
    from numpy import zeros, random, sum as np_sum, add as np_add, concatenate, \
    repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap, \
    sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide
    from gensim import matutils
    
    self=model.docvecs


    # add weights for each doc, if not already present; default to 1.0 for positive and -1.0 for negative docs
    positive = [
    (doc, 1.0) if isinstance(doc, string_types + (ndarray,) + integer_types)
    else doc for doc in positive
    ]
    negative = [
    (doc, -1.0) if isinstance(doc, string_types + (ndarray,) + integer_types)
    else doc for doc in negative
    ]

    # compute the weighted average of all docs
    all_docs, mean = set(), []
    for doc, weight in positive + negative:
        if isinstance(doc, ndarray):
            mean.append(weight * doc)
        elif doc in self.doctags or doc < self.count:
            mean.append(weight * self[doc])
        else:
            raise KeyError("doc '%s' not in trained set" % doc)
    if not mean:
        raise ValueError("cannot compute similarity with no input")
    mean = matutils.unitvec(array(mean).mean(axis=0))
    return mean


with open('moralpolitics/data/taggeddoc.txt', 'rb') as p:
    total_docs = pickle.load(p)

mflist = ['Harm', 'Fairness', 'Ingroup', 'Authority', 'Purity']

def wordvec(total_docs, party, opparty):
    totalcossim = {}
    taglist = [i[1][0][:-3] for i in total_docs]
    yearmonth = list(set([re.findall('\d+-\d+', i)[0] for i in taglist]))
    for period in yearmonth:
        print(period)
        cossim = {i: [] for i in mflist}
        docs = [i for i in total_docs if period in i[1][0]]

        model = doc2vec.Doc2Vec(size=300, alpha=0.025, min_alpha=0.025,
                        window=10, min_count=5, workers=14)
        model.build_vocab(docs)

        for epoch in range(20):
            model.train(docs)
            model.alpha -= 0.001  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        for mf in mflist:
            wordlist = []
            virtue = read_mfd('moralpolitics/data/mfd/{0}.csv'.format(mf+'Virtue'))
            vice   = read_mfd('moralpolitics/data/mfd/{0}.csv'.format(mf+'Vice'))
            for word in (virtue + vice):
                print(pos(word))
                wordlist.append(pos(word)[0])
            wordlist = unique_list([i for i in wordlist if i in model.wv.index2word])
            for mfword in wordlist:
                cossim[mf].append(
                        cosine_similarity(
                                meanvector(model, [model[party + '당']], [model[opparty + '당']]),
                                model[mfword]))
        meanse = [[np.mean(cossim[mf]),
                   float(stats.sem(cossim[mf]))] for mf in mflist]
        totalcossim[period] = [j for i in meanse for j in i]

    totalcossim = pd.DataFrame(totalcossim,
                               index=['Caremean', 'Carese',
                                      'Fairnessmean', 'Fairnessse',
                                      'Ingroupmean', 'Ingroupse',
                                      'Authoritymean', 'Authorityse',
                                      'Puritymean', 'Purityse'])
    totalcossim.to_csv('moralpolitics/result/wordvec/' + party + '.csv')

wordvec(total_docs, '민주', '새누리')
wordvec(total_docs, '새누리', '민주')
