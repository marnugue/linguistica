from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer
from glob import glob
import spacy
import string
import os
import re
# as md S
def clean(file):
    with open(file,"r") as fd:
        matches = re.finditer(regex,fd.read(),re.MULTILINE)
        with open(file,"w") as fd1:
            for _, match in enumerate(matches):    
                fd1.write(match.group())
                fd1.write("\n")
        fd1.close()
    fd.close()

def lemmatizer(text):  
  doc = nlp(text)
  return ' '.join([word.lemma_ for word in doc])

    
paths = ["../data/txt/S*","../data/txt/AS*","../data/txt/D*"]
i = 0
regex = r"((?!(\r\n|\n|\r|\t))[\s\S])+"
nlp = spacy.load("es_core_news_sm") # hay dos modelos es_dep_news_trf (accuracy, utiliza redes neuronales) y es_core_news_sm (eficiencia)
stemmer = SnowballStemmer('spanish')
datos = []
numero_documentos = 0

def cleanTexts(path,lematizer = False,stopword=True,steaming = True,extractor=False):
    
    global numero_documentos,i
        
    if extractor:
        for pathi in paths:
            for file in glob(pathi):
                clean(file)
        
    # Obtener directorio de los datos y las carpetas
    # path_datos = os.getcwd() + "/../data/dataset"
    path_datos = os.getcwd()+ "/"+ path
    carpetas = os.listdir(path_datos)

    # Cargar el contenido de los .txt en un array y etiquetarlos: Deportes = 1, Salud = 2 y Politica = 3
    for c in carpetas:

        files = os.listdir(path_datos + "/" + c)
        for f in files:
            

            # Abrir los .txt, eliminar saltos de linea y agruparlos todos en un str
            with open(path_datos + "/" + c + "/" + f, encoding="utf-8") as documento_txt:
                contenido = [linea.rstrip() for linea in documento_txt]
            texto = ""
            for part in contenido:
                texto = texto + " " + part

            if c == "Deporte":
                datos.append([texto, 1])
            elif c == "Salud":
                datos.append([texto, 2])
            elif c == "Politica":
                datos.append([texto, 3])
            numero_documentos += 1

    regex = r"[a-zA-Záéíóúñ]+"

    for instance in datos:
    
        text_tokens = re.findall(regex, instance[0])
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words("spanish")] if stopword else text_tokens
        text = " ".join(tokens_without_sw)
        doc = nlp(text)
        lematizied_text = [word.lemma_ for word in doc] if lematizer else tokens_without_sw
        steamed_words = [stemmer.stem(word) for word in lematizied_text] if steaming else lematizied_text
        instance[0] = " ".join(steamed_words)
        
    return datos
# Installer nltk package
#
#    nltk.download('stopwords')
#    nltk.download('punkt')
