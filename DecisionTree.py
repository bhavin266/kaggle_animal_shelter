#from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
#import numpy as np
#from io import StringIO
import pandas as pd

li1 = ['Name','AnimalType','AgeuponOutcome','Breed','Color','Quality','Sex','HairType','Mix','BreedGroup','MultiColored','Hour','Minute','Second','Year','Month','Day','LifeStage','DayPeriod']
li1.insert(0,'OutcomeType')

training_data = pd.read_csv("training_animals_normalized.csv")
training_data = training_data[li1]
#training_data['Date'],training_data['Time'] = zip(*training_data["DateTime"].str.split().tolist())
#training_data["Hour"],training_data["Minute"],training_data["Second"] = zip(*training_data["Time"].str.split(':').tolist())
#training_data["Year"],training_data["Month"],training_data["Date"] = zip(*training_data["Date"].str.split('-').tolist())
#del training_data['DateTime']
#del training_data['Date']
#del training_data['Time']
y_train = []
x_train = []
for row in training_data.iterrows():
	index,data = row
	li = data.tolist()
	x_train.append(li[1:])
	y_train.append(li[0])


testing_data = pd.read_csv("testing_animals_normalized.csv")
testing_data = testing_data[li1]
#testing_data['Date'],testing_data['Time'] = zip(*testing_data["DateTime"].str.split().tolist())
#testing_data["Hour"],testing_data["Minute"],testing_data["Second"] = zip(*testing_data["Time"].str.split(':').tolist())
#testing_data["Year"],testing_data["Month"],testing_data["Date"] = zip(*testing_data["Date"].str.split('-').tolist())
#del testing_data['DateTime']
#del testing_data['Date']
#del testing_data['Time']
y_test = []
x_test = []
for row in testing_data.iterrows():
	index,data = row
	li = data.tolist()
	x_test.append(li[1:])
	y_test.append(li[0])

right_count = 0
wrong_count = 0

clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
for i in range(len(y_test)):
    prediction = clf.predict([x_test[i]])[0]
    if prediction == y_test[i]:
        right_count += 1
    else:
        wrong_count += 1

print(right_count)
print(wrong_count)
print(right_count/(right_count+wrong_count))