import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer
from glob import glob
import spacy
import string
import os
import re
import matplotlib.pyplot as plt

# as md S

# Installer nltk package
#
nltk.download('stopwords')
nltk.download('punkt')


def clean(file):
    with open(file, "r") as fd:
        matches = re.finditer(regex, fd.read(), re.MULTILINE)
        with open(file, "w") as fd1:
            for _, match in enumerate(matches):
                fd1.write(match.group())
                fd1.write("\n")
        fd1.close()
    fd.close()


def lemmatizer(text):
    doc = nlp(text)
    return ' '.join([word.lemma_ for word in doc])


paths = ["../data/txt/S*", "../data/txt/AS*", "../data/txt/D*"]
i = 0
regex = r"((?!(\r\n|\n|\r|\t))[\s\S])+"
nlp = spacy.load(
    "es_core_news_sm")  # hay dos modelos es_dep_news_trf (accuracy, utiliza redes neuronales) y es_core_news_sm (eficiencia)
stemmer = SnowballStemmer('spanish')
datos = [[], [], []]
numero_documentos = 0

def media (lista):
    return sum(lista) / len(lista)

def cleanTexts(path, lematizer=False, stopword=True, steaming=True, extractor=False):
    global numero_documentos, i

    if extractor:
        for pathi in paths:
            for file in glob(pathi):
                clean(file)

    # Obtener directorio de los datos y las carpetas
    # path_datos = os.getcwd() + "/../data/dataset"
    path_datos = os.getcwd() + "/" + path
    carpetas = os.listdir(path_datos)

    # Variables para sacar estadísticas
    numero_palabras_raw = []
    numero_palabras_regex = []
    numero_palabras_stop = []
    numero_palabras_stem = []
    numero_palabras_lema = []

    # Cargar el contenido de los .txt en un array y etiquetarlos: Deportes = 1, Salud = 2 y Politica = 3
    for c in carpetas:

        print("Extrayendo carpeta " + c + "...")
        files = os.listdir(path_datos + "/" + c)
        for f in files:
            if numero_documentos % 60 == 0:
                print("Extrayendo documento " + str(numero_documentos) + "...")

            # Abrir los .txt, eliminar saltos de linea y agruparlos todos en un str
            with open(path_datos + "/" + c + "/" + f, encoding="utf-8") as documento_txt:
                contenido = [linea.rstrip() for linea in documento_txt]
            texto = ""
            for part in contenido:
                texto = texto + " " + part

            if c == "Deporte":
                datos[0].append([texto, 1])
            elif c == "Salud":
                datos[1].append([texto, 2])
            elif c == "Politica":
                datos[2].append([texto, 3])
            numero_documentos += 1

    regex = r"[a-zA-Záéíóúñ]+"
    for i in range(3):
        for index, instance in enumerate(datos[i]):

            # Contar palabras raw
            numero_palabras_raw.append(len(instance[0].split()))

            # Contar palabras regex
            if index % 10 == 0:
                print("Procesando instancia " + str(index) + "...")
            text_tokens = re.findall(regex, instance[0])
            n_palbras_documento = 0
            for palabra in text_tokens:
                n_palbras_documento += 1
            numero_palabras_regex.append(n_palbras_documento)

            # Contar palabras stop
            tokens_without_sw = [word for word in text_tokens if
                                 not word in stopwords.words("spanish")] if stopword else text_tokens
            n_palbras_documento = 0
            for palabra in tokens_without_sw:
                n_palbras_documento += 1
            numero_palabras_stop.append(n_palbras_documento)

            text = " ".join(tokens_without_sw)
            doc = nlp(text)

            # Contar palabras lema
            lematizied_text = [word.lemma_ for word in doc]
            n_palbras_documento = 0
            for palabra in lematizied_text:
                n_palbras_documento += 1
            numero_palabras_lema.append(n_palbras_documento)

            # Contar palabras stem
            steamed_words = [stemmer.stem(word) for word in tokens_without_sw]
            n_palbras_documento = 0
            for palabra in steamed_words:
                n_palbras_documento += 1
            numero_palabras_stem.append(n_palbras_documento)

            instance[0] = " ".join(steamed_words)
    eje_x = ["Raw", "Regex", "Stopwords","Lematización" ,"Stemming"]

    eje_y = [media(numero_palabras_raw), media(numero_palabras_regex), media(numero_palabras_stop), media(numero_palabras_lema), media(numero_palabras_stem)]
    print(eje_y)
    plt.rcParams.update({'font.size': 20})
    plt.bar(range(5), eje_y, edgecolor='black')
    plt.xticks(range(5), eje_x, rotation=0)
    plt.title("Número medio de palabras por documento")
    #plt.ylim(min(eje_y) - 1, max(eje_y) + 1)
    plt.show()

if __name__ == '__main__':
    data = cleanTexts("data/dataset", extractor=True)