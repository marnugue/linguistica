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
def clean(file,dir):
    with open(file,"r") as fd:
        matches = re.finditer(regex,fd.read(),re.MULTILINE)
        with open(dir,"w") as fd1:
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

if __name__ == '__main__':

        

    # for path in paths:
    #     for file in glob(path):
    #         newfile = "../data/dataset/"
    #         if i==0:
    #             newfile +="salud/"+file.split("/")[-1]
    #         else:
    #             newfile +="deporte/"+file.split("/")[-1]
    #         clean(file,newfile)
    #     i+=1
        
    # Obtener directorio de los datos y las carpetas
    path_datos = os.getcwd() + "/../data/dataset"
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

            if c == "Deportes":
                datos.append([texto, 1])
            elif c == "Salud":
                datos.append([texto, 2])
            elif c == "Politica":
                datos.append([texto, 3])
            numero_documentos += 1

    print (numero_documentos)
    # print(datos[0])
    regex = r"[a-zA-Záéíóú]+"
    text_tokens = re.findall(regex, datos[0][0])
    # print(datos[0][0].translate(str.maketrans('', '', string.punctuation)))

    emptyWords = stopwords.words("spanish")

    # for textPath in glob("../data/dataset/*/*"):
    #     with open(textPath) as fd:
    #         text = fd.read()
    #   text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words("spanish")]
    # print(tokens_without_sw)
    # break
    
    #lematización de las palabras
    text = " ".join(tokens_without_sw)
    
    doc = nlp(text)
    
    lematizied_text = [word.lemma_ for word in doc]
    
    #steamming
    steamed_words = [stemmer.stem(word) for word in lematizied_text]
    
    path = "../data/cleaned/"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+"texto.txt","w",encoding="utf-8") as fd:
        fd.write(" ".join(steamed_words))
    fd.close()
    # sacar frecuencia de las palabras y frecuencia minima 
    fdist=FreqDist(lematizied_text)
    # frecMin = min(fdist.values())
    
    # print(fdist.items())
    
        



# Installer nltk package
#
#    nltk.download('stopwords')
#    nltk.download('punkt')