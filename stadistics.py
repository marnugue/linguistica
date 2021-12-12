from collections import Counter
from glob import glob 
from utils.cleaner import cleanTexts
import pickle
import pandas as pd
path =  "data/dataset"
for folder in glob(f"{path}/*"):
    tematic = folder.split("/")[-1]
data = cleanTexts(path,lematizer = False,stopword=False,steaming = False,extractor=False)
file_pi = open('dataCleaned.obj', 'wb') 
pickle.dump(data, file_pi)
filehandler = open("dataCleaned.obj", 'rb') 


        
