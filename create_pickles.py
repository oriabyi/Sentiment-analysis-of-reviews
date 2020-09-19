import os
import dill
import numpy as np
import pandas as pd
from nltk.stem.snowball import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def get_data_for_model():
    train3 = pd.read_csv('./data/products_sentiment_train.tsv', sep='\t', names=['text', 'tonality'])\

    pos = pd.read_csv('./data/positive.csv', sep=';', names=['id', 'tdate', 'tname', 'ttext', 'ttype', 'trep', 'tfav', 'tstcount', 'tfol', 'tfrien', 'listcount', '1'])
    pos = pos[['ttext', 'ttype']]

    neg = pd.read_csv('./data/negative.csv', sep=';', names=['id', 'tdate', 'tname', 'ttext', 'ttype', 'trep', 'tfav', 'tstcount', 'tfol', 'tfrien', 'listcount', '1'])
    neg = neg[['ttext', 'ttype']]

    texts7 = np.append(pos['ttext'].values, neg['ttext'].values)
    labels7 = len(pos) * [1] + len(neg) * [0]

    texts = np.append(texts7, train3['text'].values)

    labels = labels7 + list(train3['tonality'])
    return texts, labels

texts, labels = get_data_for_model()

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(vectorizer.fit_transform(texts), labels)

os.mkdir('pickles')

with open('./pickles/vectorizer.pickle', 'wb') as f:
    dill.dump(vectorizer, f)
    print('vectorizer.pickle created!')

with open('./pickles/model.pickle',      'wb') as f:
    dill.dump(model, f)
    print('model.pickle created!')
