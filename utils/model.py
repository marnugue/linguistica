# Import de las librerias necesarias
import os
import re
import nltk
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import random
import numpy as np
import pandas as pd

# Descargar recursos necesarios
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

class Model():
    def __init__(self,data):
        self.data = data
        self.length = np.shape(data)[0]
    def train(self,train_size):
        random.shuffle(self.data)
        Train = pd.DataFrame(self.data[:int(self.length * train_size)], columns = ['Texto', 'Etiqueta'])

        X_Train = Train['Texto']
        Y_Train = Train['Etiqueta'].tolist()
        
        Test = pd.DataFrame(self.data[int(self.length * train_size):], columns = ['Texto', 'Etiqueta'])

        X_Test = Test['Texto']
        Y_Test = Test['Etiqueta'].tolist()

        tfidf_vectorizer = TfidfVectorizer()

        Train_Vectorized = tfidf_vectorizer.fit_transform(X_Train)
        Test_Vectorized = tfidf_vectorizer.transform(X_Test)

        classifier = naive_bayes.BernoulliNB()
        classifier.fit(Train_Vectorized, Y_Train)

        y_pred = classifier.predict(Test_Vectorized)

        c_matrix = confusion_matrix(y_true = Y_Test, y_pred = y_pred)
        print(c_matrix)
        
        