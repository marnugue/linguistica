import requests
import validators
import os
datasets = ["Datasets.txt","DatasetSalud.txt"]
for filename in datasets: 
    with open(f"../{filename}","r") as fd:
        i = 1
        iS = 1
        line = fd.readline()
        os.chdir("..")
        while line:
            if validators.url(line):
                r =requests.get(line.strip())
                if r.status_code!=200:
                    print(f"linea -> `{line}`")
                name = ""
                if line.find("marca.com")!=-1:
                    name = f"data/html/D{i}-D.html"
                elif line.find("as.com")!=-1:
                    name = f"data/html/D{i}-AS.html"
                elif line.find("www.mundodeportivo.com")!=-1:
                    name = f"data/html/D{i}-MD.html"
                elif line.find("neurologia.com")!=-1:
                    name = f"data/html/S{iS}.html"
                    iS+=1
                    
                if r.status_code == 200:
                    with open(name,"w") as file:
                        file.writelines(r.text)
                else:
                    name = name.split(".")[-2]
                    name +="Error.html"
                    with open(name,"w") as file:
                        file.write(line)
                file.close()  
                i+=1
            line = fd.readline()