from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import gc
from itertools import combinations

def GetDecisionTreeClassifier(x_train,y_train):
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    return clf

def GetRandomForestClassifier(x_train,y_train):
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(x_train, y_train)
    return clf

def GetAdaBoostClassifierDT(x_train,y_train):
    clf = AdaBoostClassifier(DecisionTreeClassifier(),
        n_estimators=600,
        learning_rate=1.5,
        algorithm="SAMME")
    clf = clf.fit(x_train, y_train)
    return clf

def GetAdaBoostClassifierRF(x_train,y_train):
    clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=10),
        n_estimators=600,
        learning_rate=1.5,
        algorithm="SAMME")
    clf = clf.fit(x_train, y_train)
    return clf

def TestClassifier(clf, x_test,y_test):
    right_count = 0 
    wrong_count = 0
    for i in range(len(y_test)):
        prediction = clf.predict([x_test[i]])[0]
        if prediction == y_test[i]:
            right_count += 1
        else:
            wrong_count += 1

    #print("right_count: " + str(right_count))
    #print("wrong_count: " + str(wrong_count))
    #print("confidence: " + str(right_count / (right_count + wrong_count)))
    return right_count,wrong_count

def build(df,li1):
    df1 = df.copy()
    li1.insert(0,'OutcomeType')
    df1 = df1[li1]
    if 'DateTime' in li1:
        df1['Date'],df1['Time'] = zip(*df1["DateTime"].str.split().tolist())
        df1["Hour"],df1["Minute"],df1["Second"] = zip(*df1["Time"].str.split(':').tolist())
        df1["Year"],df1["Month"],df1["Date"] = zip(*df1["Date"].str.split('-').tolist())
        del df1['DateTime']
        del df1['Date']
        del df1['Time']
    y = []
    x = []
    for row in df1.iterrows():
        index,data = row
        li = data.tolist()
        x.append(li[1:])
        y.append(li[0])
    li1.remove('OutcomeType')
    return x,y

#li1 = ['Name','AnimalType','AgeuponOutcome','Breed','Color','Quality','Sex','HairType','Mix','BreedGroup','MultiColored','Hour','Minute','Second','Year','Month','Day','LifeStage','DayPeriod']
li1 = ["Mix","AnimalType=Cat","AnimalType=Dog","BreedGroup=Herding","BreedGroup=Hound","BreedGroup=Non-Sporting","BreedGroup=Sporting","BreedGroup=Terrier","BreedGroup=Toy","BreedGroup=Unknown","BreedGroup=Working","Quality=Intact","Quality=Neutered","Quality=Spayed","Quality=Unknown","Sex=Female","Sex=Male","Sex=Unknown"]
print(li1)
training_data = pd.read_csv("training_animals_normalized_vectorized.csv")
testing_data = pd.read_csv("testing_animals_normalized_vectorized.csv")

print("Method\tRightAnswer\tWrongAnswer\tAccuracy")

x_train,y_train = build(training_data,li1)
x_test,y_test = build(testing_data,li1)

clf = GetDecisionTreeClassifier(x_train,y_train)
rc,wc = TestClassifier(clf,x_test,y_test)
print("DecisionTreeClassifier\t" + str(rc) + "\t" + str(wc) + "\t" + str(rc / (rc + wc)))
gc.collect()

clf = GetRandomForestClassifier(x_train,y_train)
rc,wc = TestClassifier(clf,x_test,y_test)
print("RandomForestClassifier\t" + str(rc) + "\t" + str(wc) + "\t" + str(rc / (rc + wc)))
gc.collect()

clf = GetAdaBoostClassifierDT(x_train,y_train)
rc,wc = TestClassifier(clf,x_test,y_test)
print("AdaBoostClassifierDT\t" + str(rc) + "\t" + str(wc) + "\t" + str(rc / (rc + wc)))
gc.collect()

clf = GetAdaBoostClassifierRF(x_train,y_train)
rc,wc = TestClassifier(clf,x_test,y_test)
print("AdaBoostClassifierRF\t" + str(rc) + "\t" + str(wc) + "\t" + str(rc / (rc + wc)))
gc.collect()

