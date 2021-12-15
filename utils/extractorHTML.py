import requests
import validators
import os
from glob import glob
# datasets = ["Datasets.txt","DatasetSalud.txt"]
class HTMLextractor(object):
    def __init__(self, path):
        self.path = path
    
    def extract(self):
        if not os.path.exists("data/html"):
            os.makedirs("data/html")
        datasets = glob(self.path+"/*")
        for filename in datasets: 
            with open(f"{filename}","r",encoding="utf-8") as fd:
                i = 1
                iS = 1
                line = fd.readline()
                while line:
                    if validators.url(line):
                        r =requests.get(line.strip())
                        if r.status_code!=200:
                            print(f"linea -> `{line}`")
                        name = ""
                        if "marca.com" in line:
                            name = f"data/html/D{i}-D.html"
                        elif "as.com" in line:
                            name = f"data/html/D{i}-AS.html"
                        elif "www.mundodeportivo.com" in line:
                            name = f"data/html/D{i}-MD.html"
                        elif line.find("neurologia.com")!=-1:
                            name = f"data/html/S{iS}.html"
                            iS+=1
                        if r.status_code == 200:
                            with open(name,"w",encoding="utf-8") as file:
                                file.writelines(r.text)
                        else:
                            name = name.split(".")[-2]
                            name +="Error.html"
                            with open(name,"w",encoding="utf-8") as file:
                                file.write(line)
                        file.close()  
                        i+=1
                    line = fd.readline()