from utils.cleaner import cleanTexts
from utils.model import Model
from utils import documentExtractor
from utils.ExtractorHTML import HTMLextractor
import pickle

models = ["bayes_bow","regresion_bow","bayes_tfidf","regresion_tfidf"]

if __name__ == '__main__':

    # Opcional si se quiere extraer los ficheros html de salud,deporte y extraer su info. En caso de descomentar la
    # linea poner en cleanTexts el atributo extractor=True
    
    # html = HTMLextractor("sources")
    # html.extract()
    # documentExtractor.extract()

    # Opciones de limpieza por defecto lematizer=False stopword=True,steaming = True
    print("\n________ Modelos stemmer ________")
    data = cleanTexts("data/dataset")
    # filehandler = open("dataCleaned.obj", 'rb') 
    # data = pickle.load(filehandler)
    # Inicializar y entrenar el modelo
    model = Model(data)
    model.train()
    # Analisis
    model.analisis(models)

    print("\n________ Modelos lematizador ________")
    data = cleanTexts("data/dataset", lematizer=True, steaming=False)
    # filehandler = open("dataCleaned.obj", 'rb')
    # data = pickle.load(filehandler)
    # Inicializar y entrenar el modelo
    model = Model(data)
    model.train()
    # Analisis
    model.analisis(models)
