import os
import csv
from os import listdir
from os.path import isfile, isdir, join, basename, dirname, splitext

def getMood(path):
    mood = basename(path).lower()
    if mood == "relax":
        mood = "relaxed"
    num = increment(mood)
    return mood, str(num)

def increment(name):
    moods[name] += 1
    return moods[name]

def renameFiles(path):
    files = sorted([f for f in listdir(path) if isfile(join(path, f))])
    if files == []:
        return False
    else:
        with open('data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            for f in files:
                title, ext = splitext(f)
                m, n = getMood(path)
                writer.writerow([m + n, title, m])
                os.rename(join(path, f), join(root, m + n + ext))
        return True

def mergeRename(path):
    dirs = getDirs(path)
    for d in dirs:
        files = sorted([f for f in listdir(join(path, d)) if isfile(join(path, d, f))])
        with open('data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            for f in files:
                title, ext = splitext(f)
                m, n = getMood(path)
                writer.writerow([m + n, title, m])
                os.rename(join(path, d, f), join(root, m + n + ext))

def getDirs(path):
    dirs = sorted([d for d in listdir(path) if isdir(join(path, d))])
    return dirs

def getPrevDir(path):
    return dirname(path)

def isTestTrain(path):
    for d in getDirs(path):
        if d.lower() == "test" or d.lower() == "train":
            return True
    return False

def renameRecursive(path):
    for d in getDirs(path):
        if not renameFiles(join(path, d)):
            if isTestTrain(join(path, d)):
                print("Merging test and train.")
                mergeRename(join (path, d))
            renameRecursive(join(path, d))

def createCSV():
    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        fields = ["id", "name", "mood"]
        writer.writerow(fields)

def main():
    global root
    root = input("Enter the name of the folder that the data is located in: ")
    createCSV()
    renameRecursive(root)

moods = {"angry": 0, "happy": 0, "relaxed": 0, "sad": 0}
root = None

if __name__ ==  "__main__":
    main()