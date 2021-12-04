import os
from glob import glob
os.chdir("../data/dataset/deporte")
for path in glob("*"):
    num = path.split("-")[0]
    print(f"{num}")
    os.rename(path,f"{num}.txt")