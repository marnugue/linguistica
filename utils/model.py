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
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity


class Model():
    def __init__(self,data):
        self.data = data
        self.length = np.shape(data)[0]
    def cos_similarity(self, x_train, x_test, y_train):
        y_test = []
        for test_instance in x_test:
            candidate_similarity = 0
            candidate_label = 0
            for train_instance, label in zip(x_train, y_train):
                similarity = cosine_similarity(test_instance, train_instance)
                if similarity > candidate_similarity:
                    candidate_label = label
                    candidate_similarity = similarity
            y_test.append(candidate_label)
        return y_test
    def train(self,train_size):
        print("Entrenando el modelo...")
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

        '''print(np.shape(Train_Vectorized))
        print(np.shape(Test_Vectorized))

        classifier = naive_bayes.BernoulliNB()
        classifier.fit(Train_Vectorized, Y_Train)

        y_pred = classifier.predict(Test_Vectorized)
        y_pred_prob = classifier.predict_proba(Test_Vectorized)
        print(y_pred_prob)'''

        y_pred = self.cos_similarity(Train_Vectorized, Test_Vectorized, Y_Train)

        c_matrix = confusion_matrix(y_true = Y_Test, y_pred = y_pred)
        print(c_matrix)
        print(classification_report(y_true = Y_Test, y_pred = y_pred, target_names = ['Deporte', 'Salud', 'Política']))