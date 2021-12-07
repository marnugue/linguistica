from utils.cleaner import cleanTexts
from utils.model import Model
from utils import documentExtractor
from utils.extractorHTML import HTMLextractor

if __name__ == '__main__':
    # opcional si se quiere extraer los ficheros html de salud,deporte y extraer su info. En caso de descomentar la linea poner en cleanTexts el atributo extractor=True
    # html = HTMLextractor("sources")
    # html.extract()
    # documentExtractor.extract()
    # opciones de limpieza por defecto lematizer=False stopword=True,steaming = True
    data = cleanTexts("data/dataset",extractor=True)
    model = Model(data)
    model.train(train_size=0.7)