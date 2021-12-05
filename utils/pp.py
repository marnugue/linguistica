import spacy
nlp = spacy.load("es_core_news_sm")

text = 'Los gatos y los perros juegan jugaban jugado juntos en el patio de pierde pierden su casa'
doc = nlp(text)
for word in doc:
    print(word.lemma_,word)