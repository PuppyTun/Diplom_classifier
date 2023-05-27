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

df = pd.read_csv('news_lemmatized_2023-05-27_15_56_38.888900.csv')

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

# loaded_model = pickle.load(open("SGDClassifier.sav", 'rb'))
loaded_model = pickle.load(open("LinearSVC_2023-05-27_15_56_38.888900.sav", 'rb'))

class cWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(cWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.Button1.clicked.connect(self.Button1Clicked)
        self.ui.Button2.clicked.connect(self.Button2Clicked)

    def Button1Clicked(self):

        input_Text = self.ui.TextEdit.toPlainText()

        input_Text = re.sub(r'[^\w\s-]', '', input_Text)
        input_Text = input_Text.replace('_', ' ')
        input_Text = input_Text.replace('-', ' ')

        words_text = re.sub(r'[^а-яА-Яa-zA-Z]', ' ', input_Text)
        words_text = re.sub(r'\s{2,}', ' ', words_text)

        if len(words_text) == 0:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle('Ошибка')
            msg.setText('Исходное сообщение должно содержать ' + \
                        'по крайней мере один символ!')
            # msg.setDetailedText('')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return

        ar_words = words_text.split()
        ar_english_words = re.findall(r'[a-zA-Z]+', words_text)

        procent_english_words = len(ar_english_words) / (len(ar_words) / 100)

        if (procent_english_words > 50.0):
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle('Ошибка')
            msg.setText('Введите текст на русском языке!\nВведенный текст в основном на английском.')
            # msg.setDetailedText('')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return

        with open('DictionaryRussianWords.txt', 'r', encoding="utf-8") as f:
            rus_dict_word = f.readlines()

        ar_russian_words = re.findall(r'[а-яА-Я]+', words_text)
        # ar_rus_words = russian_words.split()
        count_all_words = len(ar_russian_words)
        count_find_words = 0
        for word in ar_russian_words:
            if (word.lower() + '\n') in rus_dict_word:
                count_find_words += 1

        procent_true_words = count_find_words / (count_all_words / 100)

        if (procent_true_words < 50.0):
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle('Ошибка')
            msg.setText('Не удалось распознать слова в тексте!')
            # msg.setDetailedText('')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return

        # sTag = classificator(input_Text)
        sTag = classificator(words_text)
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
