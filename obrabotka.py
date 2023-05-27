import pandas as pd
import numpy as np
import json
import re
from warnings import filterwarnings
filterwarnings('ignore')
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
#
#
df = pd.read_csv('news_lemmatized_2023-05-27_15_56_38.888900.csv')
#
# label_encoder = LabelEncoder()
# corpus_encoded = label_encoder.fit_transform(df.tags)
# #
# tags = []
# for i in range(len(corpus_encoded)):
#   tags.append((df.tags[i], corpus_encoded[i]))
#
# df.tags = corpus_encoded
#
# dict_tags = dict()
#
# for i in range(len(tags)):
#   dict_tags[tags[i][1]]=tags[i][0]

# label_encoder = LabelEncoder()
# corpus_encoded = label_encoder.fit_transform(df.tags)
#
# df.tags = corpus_encoded
#
# answer = []
# def get_title_corpus(title):
#     answer.append(' '.join(re.findall(r'[а-я-]+', title)))
#
# for title in df.title:
#     get_title_corpus(title)
#
#
# tfidf = TfidfVectorizer()
# title_tfidf = tfidf.fit_transform(answer)
# feature_names = tfidf.get_feature_names_out()
# corpus_index = [n for n in answer]
#
# df_tfidf = pd.DataFrame(title_tfidf.todense(), index=corpus_index, columns=feature_names)
#
# x = df_tfidf
# y = df.tags
#
# # x.index = df.index
# #
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#
# loaded_model = pickle.load(open("SGDClassifier.sav", 'rb'))
#
# prediction = ['dhrrhdrjjrd rshhsrsr']
#
# pred = loaded_model.predict(tfidf.transform(prediction))
# print(dict_tags[pred[0]])

# with open('DictionaryRussianWords.txt', 'r', encoding="utf-8") as f:
#     rus_dict_word = f.readlines()

'''
_________________________________________'''
label_encoder = LabelEncoder()
corpus_encoded = label_encoder.fit_transform(df.tags)

tags = []
for i in range(len(corpus_encoded)):
  tags.append((df.tags[i], corpus_encoded[i]))

df.tags = corpus_encoded

dict_tags = dict()

for i in range(len(tags)):
  dict_tags[tags[i][1]]=tags[i][0]

label_encoder = LabelEncoder()
corpus_encoded = label_encoder.fit_transform(df.tags)

df.tags = corpus_encoded

answer = []
def get_title_corpus(title):
    answer.append(' '.join(re.findall(r'[а-я-]+', title)))

for title in df.title:
    get_title_corpus(title)

tfidf = TfidfVectorizer()
title_tfidf = tfidf.fit_transform(answer)
feature_names = tfidf.get_feature_names_out()
corpus_index = [n for n in answer]

df_tfidf = pd.DataFrame(title_tfidf.todense(), index=corpus_index, columns=feature_names)

x = df_tfidf
y = df.tags

x.index = df.index

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

loaded_model = pickle.load(open("LinearSVC_2023-05-27_15_56_38.888900.sav", 'rb'))

prediction = ['Спутник «Метеор-М» № 2-3 доставили на Восточный.']

pred = loaded_model.predict(tfidf.transform(prediction))
# pred[0]
print(dict_tags[pred[0]])

score = accuracy_score(y_train, pred)
print(score)
