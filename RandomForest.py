from sklearn.ensemble import RandomForestClassifier
import numpy as np
from io import StringIO
import pandas as pd

training_data = pd.read_csv("training_animals_normalized.csv")
training_data = training_data[['OutcomeType','AnimalType','AgeuponOutcome','Breed','Color','Sex','Quality']]
y_train = []
x_train = []
for row in training_data.iterrows():
	index,data = row
	li = data.tolist()
	x_train.append(li[1:])
	y_train.append(li[0])

testing_data = pd.read_csv("testing_animals_normalized.csv")
testing_data = testing_data[['OutcomeType','AnimalType','AgeuponOutcome','Breed','Color','Sex','Quality']]
y_test = []
x_test = []
for row in testing_data.iterrows():
	index,data = row
	li = data.tolist()
	x_test.append(li[1:])
	y_test.append(li[0])

right_count = 0
wrong_count = 0

clf = RandomForestClassifier(n_estimators=600)
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