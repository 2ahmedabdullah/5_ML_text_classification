# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:12:57 2019

@author: 1449486
"""


from pyexcel_ods import get_data
data = get_data("oblign-non.ods")

c=data['Sheet1']


import pandas as pd
df=pd.DataFrame(c)
df=df.dropna()


X=df[df.columns[0]]
y=df[df.columns[1]]


from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)


from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()

x_train_tf=count_vect.fit_transform(X_train)
x_train_tf.shape



from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transform=TfidfTransformer()
x_train_tfidf=tfidf_transform.fit_transform(x_train_tf)
x_train_tfidf.shape

x_test_tf=count_vect.transform(X_test)
x_test_tfidf=tfidf_transform.transform(x_test_tf)


#......................different models...................................

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB


from sklearn.model_selection import cross_val_score
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0)
]

from sklearn import metrics
import numpy as np

clf=[]
prdd=[]
c_m=[]
acc=[]
precision=[]
recall=[]
fscore=[]

for i in range (0,len(models)):
    c=models[i].fit(x_train_tfidf,Y_train)
    p=c.predict(x_test_tfidf)
    cmm=metrics.confusion_matrix(Y_test, p)
    
    clf.append(c)
    prdd.append(p)
    precision.append(round(pr,2))
    recall.append(round(re,2))
    fscore.append(round(fs,2))
    c_m.append(cmm)
    acc.append(round(np.mean(prdd[i] == Y_test),2))



predict=list(zip(prdd[0],prdd[1],prdd[2],prdd[3],Y_test))
predict=pd.DataFrame(predict)
predict=pd.DataFrame(predict.values, columns = ["RandomForest", 
        "SVM","MultiNB","ComplementNB","LogisticRegression"])

    
    
labels = ["RandomForest", "SVM","MultiNB","ComplementNB","LogisticRegression"]

summary=pd.DataFrame()
summary['Classifier']=labels
summary['Accuracy']=acc
summary['Precision']=precision
summary['Recall']=recall 
summary['Fscore']=fscore




#................................word2vec.....................................


import gensim
# let X be a list of tokenized texts (i.e. list of lists of tokens)
model = gensim.models.Word2Vec(X, size=100)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))



from gensim.models import Word2Vec
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]

# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)





from gensim.models import KeyedVectors
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)





from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.txt'
word2vec_output_file = 'word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)














