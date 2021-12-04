from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from glob import glob
emptyWords = stopwords.words("spanish")

for textPath in glob("../data/dataset/*/*"):
    with open(textPath) as fd:
        text = fd.read()
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        print(tokens_without_sw)
        break

# Installer nltk package
#
#    nltk.download('stopwords')
#    nltk.download('punkt')
