# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 00:26:47 2017

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

total_docs = [i for i in total_docs if '민주2012-' in i[1][0]]

model = doc2vec.Doc2Vec(size=300, alpha=0.025, min_alpha=0.025,
                        window=10, min_count=20, workers=14)
model.build_vocab(total_docs)

for epoch in range(20):
    model.train(total_docs)
    model.alpha -= 0.001  # decrease the learning rate

    model.min_alpha = model.alpha  # fix the learning rate, no decay

# model.init_sims(replace=True)
# 메모리 세이브용이나 infer_vector를 위해서는 쓰면 안 됨
model.save('moralpolitics/data/doc2vec')
# model = doc2vec.Doc2Vec.load('moralpolitics/data/doc2vec')

MFDictionary = ['HarmVirtue', 'HarmVice', 'FairnessVirtue', 'FairnessVice',
                'IngroupVirtue', 'IngroupVice', 'AuthorityVirtue',
                'AuthorityVice', 'PurityVirtue', 'PurityVice']

for i in MFDictionary:
    wordlist = []
    for j in read_mfd('moralpolitics/data/mfd/{0}.csv'.format(i)):
        print(pos(j))
        Morp = pos(j)[0]
        wordlist.append(Morp)
    wordlist = [i for i in wordlist if i in model.wv.index2word]
    exec("%s=%s" % (i, unique_list(wordlist)))

mflist = ['Harm', 'Fairness', 'Ingroup', 'Authority', 'Purity']

for mf in mflist:
    exec("%s = meanvector(model, "
         "[model[i] for i in eval(mf + 'Virtue')]+[model[i] for i in eval(mf + 'Vice')], "
         "[])" % mf)

def docvecsim(model, party, opparty):

    taglist = [i for i in model.docvecs.doctags]
    yearmonth = list(set([re.findall('\d+-\d+', i)[0] for i in taglist]))
    totalcossim = {}
    for period in yearmonth:
        doctags = [i for i in taglist if party + period in i]
        cossim = {i: [] for i in mflist}
        for doctag in doctags:
            print(doctag)
            docvec = meanvector(model,
                                [model.docvecs[doctag],
                                 model[opparty + '당']],
                                [model[party + '당']])
            for mf in mflist:
                cossim[mf].append(cosine_similarity(
                    eval(mf), docvec))
        meanse = [[np.mean(cossim[mf]),
                   float(stats.sem(cossim[mf]))] for mf in mflist]
        totalcossim[period] = [j for i in meanse for j in i]

    totalcossim = pd.DataFrame(totalcossim,
                               index=['Caremean', 'Carese',
                                      'Fairnessmean', 'Fairnessse',
                                      'Ingroupmean', 'Ingroupse',
                                      'Authoritymean', 'Authorityse',
                                      'Puritymean', 'Purityse'])
    totalcossim.to_csv('moralpolitics/result/docvec/' + party + '.csv')

docvecsim(model, '민주', '새누리')
docvecsim(model, '새누리', '민주')
