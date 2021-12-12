from utils.cleaner import cleanTexts
from utils.model import Model
from utils import documentExtractor
from utils.extractorHTML import HTMLextractor
import pickle
if __name__ == '__main__':

    # Opcional si se quiere extraer los ficheros html de salud,deporte y extraer su info. En caso de descomentar la
    # linea poner en cleanTexts el atributo extractor=True

    # html = HTMLextractor("sources")
    # html.extract()
    # documentExtractor.extract()

    # Opciones de limpieza por defecto lematizer=False stopword=True,steaming = True
    data = cleanTexts("data/dataset",extractor=True)
    # filehandler = open("dataCleaned.obj", 'rb') 
    # data = pickle.load(filehandler)
    # Inicializar y entrenar el modelo
    model = Model(data)
    model.train()