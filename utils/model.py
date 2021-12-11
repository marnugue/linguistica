# Import de las librerias necesarias
import random
import numpy as np
import pandas as pd

# Descargar recursos necesarios
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

class Model():

    def __init__(self,data):
        self.data = data
        self.length = np.shape(data)[0]

    # Método para preparar subconjuntos de entrenamiento y test
    def prep_dataset(self):
        datos_deporte = self.data[0]
        datos_salud =  self.data[1]
        datos_politica = self.data[2]
        random.shuffle(datos_deporte)
        random.shuffle(datos_salud)
        random.shuffle(datos_politica)

        Train_list = []
        Test_list = []

        for i in range(60):
            Train_list.append(datos_deporte[i])
            Train_list.append(datos_salud[i])
            Train_list.append(datos_politica[i])
        for i in range(60,90):
            Test_list.append(datos_deporte[i])
            Test_list.append(datos_salud[i])
            Test_list.append(datos_politica[i])

        Train = pd.DataFrame(Train_list, columns=['Texto', 'Etiqueta'])

        Test = pd.DataFrame(Test_list, columns=['Texto', 'Etiqueta'])

        x_train = Train['Texto']
        y_train = Train['Etiqueta'].tolist()
        x_test = Test['Texto']
        y_test = Test['Etiqueta'].tolist()
        #print(np.shape(x_train), np.shape(x_test))
        #print (x_train)

        return x_train, y_train, x_test, y_test


    # Método para obtener predicciones clasificador similitud del coseno
    def cos_similarity_classification(self, x_train, x_test, y_train):
        print("Entrenando modelo cos similarity...\n")
        y_test = []
        y_similarity = []
        for test_instance in x_test:
            candidate_similarity = 0
            candidate_label = 0
            for train_instance, label in zip(x_train, y_train):
                similarity = cosine_similarity(test_instance, train_instance)
                
                if similarity > candidate_similarity:
                    candidate_label = label
                    candidate_similarity = similarity
            y_test.append(candidate_label)
            y_similarity.append(candidate_similarity)

        return (y_test,y_similarity)

    # Método para obtener predicciones clasificador Bayes. Devuelve predicción de etiquetas y probabilidades
    def bayes_classification(self, x_train, x_test, y_train):
        print("Entrenando modelo Bayes...\n")
        classifier = naive_bayes.BernoulliNB()
        classifier.fit(x_train, y_train)
        y_pred_prob = classifier.predict_proba(x_test)
        y_pred = classifier.predict(x_test)
        return (y_pred, y_pred_prob)

    def logistic_regresion_clasification(self, x_train, x_test, y_train):
        print("Entrenando modelo Regresion Logistica...\n")
        classifier = LogisticRegression()
        classifier.fit(x_train, y_train)
        y_pred_prob = classifier.predict_proba(x_test)
        y_pred = classifier.predict(x_test)
        return (y_pred, y_pred_prob)

    def train(self):
        print("Entrenando los modelos...\n")
        x_train, y_train, x_test, y_test = self.prep_dataset()

        bow_vectorizer = CountVectorizer()
        Train_Vectorized_bow = bow_vectorizer.fit_transform(x_train)
        Test_Vectorized_bow = bow_vectorizer.transform(x_test)

        tfidf_vectorizer = TfidfVectorizer()
        Train_Vectorized_tfidf = tfidf_vectorizer.fit_transform(x_train)
        Test_Vectorized_tfidf = tfidf_vectorizer.transform(x_test)

        print("\n________ Modelo Similitud Coseno ________")
        y_pred,similarity = self.cos_similarity_classification(Train_Vectorized_tfidf, Test_Vectorized_tfidf, y_train)
        c_matrix = confusion_matrix(y_true = y_test, y_pred = y_pred)
        print(c_matrix)
        print(classification_report(y_true = y_test, y_pred = y_pred, target_names = ['Deporte', 'Salud', 'Política']))

        print("\n________ Modelo Bayes ________")
        y_pred,y_probs = self.bayes_classification(Train_Vectorized_tfidf, Test_Vectorized_tfidf, y_train)
        c_matrix = confusion_matrix(y_true=y_test, y_pred=np.argmax(y_probs,axis=1)+1)
        print(c_matrix)
        print(classification_report(y_true=y_test, y_pred=np.argmax(y_probs,axis=1)+1, target_names=['Deporte', 'Salud', 'Política']))

        print("\n________ Modelo Regresión Logística ________")
        y_pred,y_probs = self.logistic_regresion_clasification(Train_Vectorized_tfidf, Test_Vectorized_tfidf, y_train)
        c_matrix = confusion_matrix(y_true=y_test, y_pred=np.argmax(y_probs,axis=1)+1)
        print(c_matrix)
        print(classification_report(y_true=y_test, y_pred=np.argmax(y_probs,axis=1)+1, target_names=['Deporte', 'Salud', 'Política']))