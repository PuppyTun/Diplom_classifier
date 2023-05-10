import pandas as pd
import numpy as np
import json
import re
from warnings import filterwarnings
filterwarnings('ignore')
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('news_lemmatized.csv')

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

loaded_model = pickle.load(open("SGDClassifier.sav", 'rb'))

prediction = ['dhrrhdrjjrd rshhsrsr']

pred = loaded_model.predict(tfidf.transform(prediction))
# print(loaded_model.score(tfidf.transform(prediction), y_test))
print(dict_tags[pred[0]])