'''
iNum - integer
arNum - массив
sNum - строка
dNum - дробь
obNum - объект
htmlNum jsonNum
'''


from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

from Classifier import Ui_MainWindow
import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

def classificator(text):
    sResult = ''
    try:
        arText = [text]
        result_tiidf = tfidf.transform(arText)
        pred = loaded_model.predict(result_tiidf)
        sResult = dict_tags[pred[0]]
    except Exception:
        sResult = 'Извините, произошла ошибка'
    return sResult

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

class cWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(cWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.Button1.clicked.connect(self.Button1Clicked)
        self.ui.Button2.clicked.connect(self.Button2Clicked)

    def Button1Clicked(self):

        input_Text = self.ui.TextEdit.toPlainText()

        pattern = re.compile("[A-Za-z]+")
        matches = pattern.findall(input_Text)

        if matches:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle('Информация')
            msg.setText('English!')
            msg.setDetailedText('English')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return


        n_input_Text = len(input_Text)
        if n_input_Text == 0:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle('Информация')
            msg.setText('Исходное сообщение должно содержать ' + \
                        'хотя бы 1 символ!')
            msg.setDetailedText('Длина исходного сообщения ' + \
                                'должна быть больше 0')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return

        sTag = classificator(input_Text)
        self.ui.LineEdit.setText(sTag)
        pass

    def Button2Clicked(self):
        self.ui.TextEdit.clear()
        self.ui.LineEdit.clear()
        pass


# -*- Main -*-
app = QtWidgets.QApplication([])
application = cWindow()
application.show()
sys.exit(app.exec())
