# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:50:19 2017

@author: Changdong
"""
# Remove document tags

with open('moralpolitics/data/taggeddocmonthly.txt', 'rb') as p:
    total_docs = pickle.load(p)

model = doc2vec.Doc2Vec(size=300, alpha=0.025, min_alpha=0.025,
                        window=10, min_count=20, workers=14)
model.build_vocab(total_docs)

for epoch in range(20):
    model.train(total_docs)
    model.alpha -= 0.001  # decrease the learning rate

    model.min_alpha = model.alpha  # fix the learning rate, no decay

# model.init_sims(replace=True)
# 메모리 세이브용이나 infer_vector를 위해서는 쓰면 안 됨
model.save('moralpolitics/data/doc2vecmonthly')
model = doc2vec.Doc2Vec.load('moralpolitics/data/doc2vecmonthly')


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
'''
mflist = ['Harm', 'Fairness', 'Ingroup', 'Authority', 'Purity']

for mf in mflist:
    exec("%s = meanvector(model, "
         "[model[i] for i in eval(mf + 'Virtue')] + [model[i] for i in eval(mf + 'Vice')], "
         "[])" % mf)
'''
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
from konlpy.tag import Twitter
from gensim.models import doc2vec
import re
import pickle
import pandas as pd

model = doc2vec.Doc2Vec.load('moralpolitics/data/doc2vecmonthly')

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

colors = np.hstack([np.repeat(['#FFFF00'], len(HarmVirtue)),
                   np.repeat(['#CCCC00'], len(HarmVice)),
                   np.repeat(['#FF3300'], len(FairnessVirtue)),
                   np.repeat(['#FF0000'], len(FairnessVice)),
                   np.repeat(['#00CC33'], len(IngroupVirtue)),
                   np.repeat(['#006633'], len(IngroupVice)),
                   np.repeat(['#6699FF'], len(AuthorityVirtue)),
                   np.repeat(['#0033FF'], len(AuthorityVice)),
                   np.repeat(['#9900CC'], len(PurityVirtue)),
                   np.repeat(['#6600FF'], len(PurityVice))])
    


textplot = tsne.fit_transform(mfvectors)

#K-means Clustering은 직관적으로 MFT의 각 기준과 맞지는 않음. 결합시킬 방법 있을까?
'''
kmeans = KMeans()
kmeans.fit(textplot)
colors = kmeans.predict(textplot)
'''

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


