import requests
import validators
# r =requests.get('https://www.marca.com/futbol/balon-oro/2021/12/01/61a76931e2704e6f998b457c.html')
# with open("data/prueba.html","w") as file:
#     file.writelines(r.text)
with open("Datasets.txt","r") as fd:
    i = 1
    line = fd.readline()
    while line:
        if validators.url(line):
            r =requests.get(line)
            if r.status_code == 200:
                name = ""
                if line.find("marca.com")!=-1:
                    name = f"data/html/D{i}-D.html"
                elif line.find("as.com")!=-1:
                    name = f"data/html/D{i}-AS.html"
                else:
                    name = f"data/html/D{i}-MD.html"

                with open(name,"w") as file:
                    file.writelines(r.text)
            else:
                name = f"data/html/D{i}-Error.html"
                with open(name,"w") as file:
                    file.write(line)
            file.close()  
            i+=1
        line = fd.readline()