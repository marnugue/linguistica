import os 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
DF = {}
#bucle s
with open("../data/cleaned/texto.txt","r",encoding="utf-8") as fd:
    text = fd.read()
    text = text.split(" ")
fd.close()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text)
query = "encontr solucion ir rap pas pas siempr"
query_vect = vectorizer.transform([query])
results = cosine_similarity(X,query_vect)
print(len(results),len(query),len(text))