import re 
from glob import glob
from bs4 import BeautifulSoup
import os 
"""
Tener en cuenta:
- figcaption
- span
- txt-btn
- a
- section
"""

def extract():
    if not os.path.exists("data/dataset"):
        os.makedirs("data/dataset")
        os.makedirs("data/dataset/Deporte")
        os.makedirs("data/dataset/Salud")
    
    files = glob("data/html/*")
    iD = 1
    iS = 1
    for file in files:
        with open(file, 'r') as f:
            contents = f.read()
        soup = BeautifulSoup(contents,features="html.parser")
        file = file.split("/")[-1]
        print(file)
        newFormat = ".txt"
        if "AS.html" in file:
            newFormat = "D" +str(iD) + newFormat
            iD+=1
            print(f"\r{file}",end="",flush=True)
            tiltle = soup.find("h1",{"class":"art-headline"})
            description = soup.find("meta",  attrs={'name': 'Description'})
            soup = soup.find("div", {"id": "cuerpo_noticia"})
            # eliminamos imagenes
            for tag in soup.find_all("figcaption"):
                tag.replaceWith("")
            # eliminamos botones
            for tag in soup.find_all("span",{"class":"txt-btn"}):
                tag.replaceWith("")
            # eliminamos enlaces a
            for tag in soup.find_all("a"):
                tag.replaceWith("")
            # eliminamos secciones (identifican contenido interactivo nada mas )
            for tag in soup.find_all("section"):
                tag.replaceWith("")
            with open(f"data/dataset/Deporte/{newFormat}","w") as fd:
                fd.writelines(tiltle.get_text())
                fd.write("\n")
                fd.writelines(description["content"])
                fd.writelines(soup.get_text())
            fd.close()
            pass
        elif "-D.html" in file:
            newFormat = "D" +str(iD) + newFormat
            iD+=1
            # cuerpo de la noticia está en div ue-c-article__body
            # header está en div ue-c-article__header
            # quitar etiqueta div de deporte ue-c-article__kicker-container 
            
            header = soup.find("div",{"class":"ue-c-article__header"})
            description = header.find("p",{"class":"ue-c-article__standfirst"})
            header = header.find("h1",{"class":"ue-c-article__headline"})
            
            
            soup = soup.find("div", {"class": "ue-c-article__body"})
            # eliminamos imagenes
            try:
                for tag in soup.find_all("figcaption"):
                    tag.replaceWith("")
            except:
                pass
            # eliminamos botones
            try:
                for tag in soup.find_all("span",{"class":"txt-btn"}):
                    tag.replaceWith("")
            except:
                pass
            
            # eliminamos enlaces a
            try:
                for tag in soup.find_all("a"):
                    tag.replaceWith("")
            except:
                pass
        
            with open(f"data/dataset/Deporte/{newFormat}","w") as fd:
                fd.writelines(header.get_text())
                fd.write("\n")
                if description:    
                    fd.writelines(description.get_text())
                    fd.write("\n")
                fd.writelines(soup.get_text())
            fd.close()
        elif "-MD.html" in file:
            newFormat = "D" +str(iD) + newFormat
            iD+=1
            # title 
            # descripcion -> epigraph-container
            # body -> classpigraph-container 
            # eliminar -> span figure a 
        
            header = soup.find("title")
            description = soup.find("div",{"class":"epigraph-container"})
            soup = soup.find("div", {"class": "article-content"})
            # eliminamos imagenes
            try:
                for tag in soup.find_all("figure"):
                    tag.replaceWith("")
            except:
                pass
            # eliminamos botones
            try:
                for tag in soup.find_all("span"):
                    tag.replaceWith("")
            except:
                pass
            
            # eliminamos enlaces a
            try:
                for tag in soup.find_all("a"):
                    tag.replaceWith("")
            except:
                pass
            # eliminamos autor
            try:
                for tag in soup.find_all("div",{"class":"article-info"}):
                    tag.replaceWith("")
            except:
                pass
            # eliminamos tags
            try:
                for tag in soup.find_all("div",{"class":"tags-container"}):
                    tag.replaceWith("")
            except:
                pass
        
            # eliminamos botones
            try:
                for tag in soup.find_all("div",{"class":"spotim-open"}):
                    tag.replaceWith("")
            except :
                pass
            with open(f"data/dataset/Deporte/{newFormat}","w") as fd:
                fd.writelines(header.get_text())
                fd.write("\n")
                if description:    
                    fd.writelines(description.get_text())
                    fd.write("\n")
                fd.writelines(soup.get_text())
            fd.close()
            pass
        elif file[0]=="S":
            newFormat = "S" +str(iS) + newFormat
            header = soup.find("h1",{"class":"titulo"})
            soup = soup.find("span", {"class": "texto"})
            # no hace falta ninguna limpieza en el texto html
            with open(f"data/dataset/Salud/{newFormat}","w") as fd:
                fd.writelines(header.get_text())
                fd.write("\n")
                fd.writelines(soup.get_text())
            fd.close()
            iS+=1
        
        