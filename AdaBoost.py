from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

training_data = pd.read_csv("training_animals_normalized.csv")
training_data = training_data[['OutcomeType','AnimalType','AgeuponOutcome','Breed','Mix','Color','Sex','Quality']]
y_train = []
x_train = []
for row in training_data.iterrows():
	index,data = row
	li = data.tolist()
	x_train.append(li[1:])
	y_train.append(li[0])

testing_data = pd.read_csv("testing_animals_normalized.csv")
testing_data = testing_data[['OutcomeType','AnimalType','AgeuponOutcome','Breed','Mix','Color','Sex','Quality']]
y_test = []
x_test = []
for row in testing_data.iterrows():
	index,data = row
	li = data.tolist()
	x_test.append(li[1:])
	y_test.append(li[0])


clf = AdaBoostClassifier(DecisionTreeClassifier(),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")

clf.fit(x_train,y_train)

right_count = 0 
wrong_count = 0
for i in range(len(y_test)):
    prediction = clf.predict([x_test[i]])[0]
    if prediction == y_test[i]:
        right_count += 1
    else:
        wrong_count += 1

print(right_count)
print(wrong_count)
print(right_count / (right_count + wrong_count))