from glob import glob
import re
# as md S
def clean(file,dir):
    with open(file,"r") as fd:
        matches = re.finditer(regex,fd.read(),re.MULTILINE)
        with open(dir,"w") as fd1:
            for _, match in enumerate(matches):    
                fd1.write(match.group())
                fd1.write("\n")
        fd1.close()
    fd.close()
    
paths = ["../data/txt/S*","../data/txt/AS*","../data/txt/D*"]
i = 0
regex = r"((?!(\r\n|\n|\r|\t))[\s\S])+"

for path in paths:
    for file in glob(path):
        newfile = "../data/dataset/"
        if i==0:
            newfile +="salud/"+file.split("/")[-1]
        else:
            newfile +="deporte/"+file.split("/")[-1]
        clean(file,newfile)
    i+=1
