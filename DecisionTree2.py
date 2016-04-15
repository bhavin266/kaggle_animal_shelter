from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from io import StringIO
import pandas as pd

def toDays(age):
    if type(age) != float:
        #print(age)
        age_digit = age.split(" ")[0]
        if "week" in age:
            age = int(age_digit) * 7
        elif 'year' in age:
            age = int(age_digit) * 365
        elif 'month' in age:
            age = int(age_digit) * 30
        else:
            age = int(age_digit)
    else:
        age = 0
    return age + 1

def predictTrainingData(data,toTrain,toPredict):
    dataC = data.copy()

    select = toTrain[:]
    select.append(toPredict)

    trainData = dataC[select].dropna()
    
    if 'AgeuponOutcome' in toTrain:
        days = lambda x : toDays(x)
        dataC['AgeuponOutcome'] = dataC['AgeuponOutcome'].apply(days)
    pass

data = pd.read_csv("train_animal.csv")

d1 = data.isnull().sum().to_dict()
del d1['AnimalID']
del d1['OutcomeSubtype']
del d1['DateTime']

useToPredict = []
predict = []
for key in sorted(d1,key=d1.get,reverse=True):
	if d1[key] == 0:
		useToPredict.append(key)
	else:
		predict.append(key)

while len(predict) > 0:
    toPredict = predict.pop()
    predictTrainingData(data,toTrain,toPredict)