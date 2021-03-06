import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd

train_X = []
train_Y = []
def generate_train_X(dataframe):

    for row in dataframe.iterrows():
        
        train_X.append(list(row[-1]))

def generate_train_Y(dataframe):

    for row in dataframe.iterrows():
        train_Y.extend(list(row[-1]))

def perform_naivebase(train_X, train_Y, dataframe_test, dataframe_test_modeloutput):
    predict_list = []
    actual_list = []
    total_correct = 0
    X = np.array(train_X).astype(np.float)
    Y = np.array(train_Y).astype(np.float)
    X.reshape(len(X), len(li1))
    print(X)
    clf = GaussianNB()
    clf.fit(X, Y)
    for index in dataframe_test.iterrows():
        predict_list.append(clf.predict([index[-1]]))

    for index in dataframe_test_modeloutput.iterrows():
        actual_list.extend(list(index[-1]))

    for i in range(len(predict_list) -1):
        if predict_list[i] == actual_list[i]:
            total_correct += 1

    print(float(total_correct/len(predict_list)) * 100)

li1 = ['Name','AnimalType','AgeuponOutcome','Breed','Color','Quality','Sex','HairType','Mix','BreedGroup','MultiColored','Hour','Minute','Second','Year','Month','Day','LifeStage','DayPeriod']
def main():

    dataframe = pd.read_csv('training_animals_normalized.csv')
    dataframe1 = dataframe[li1]
    # dataframe1 = dataframe[['OutcomeType','AnimalType','AgeuponOutcome']]
    dataframe1= dataframe1.fillna(-1)
    generate_train_X(dataframe1)
    dataframe2 = dataframe[['OutcomeType']]
    generate_train_Y(dataframe2)
    dataframe_test = pd.read_csv('testing_animals_normalized.csv')
    dataframe_test_1 = dataframe_test[li1]
    # dataframe_test_1 = dataframe_test[['OutcomeType','AnimalType','AgeuponOutcome']]
    dataframe_test_modeloutput = dataframe_test[['OutcomeType']]
    dataframe_test_modeloutput = dataframe_test_modeloutput.fillna(-1)
    dataframe_test_1 = dataframe_test_1.fillna(-1)
    print(train_X)
    perform_naivebase(train_X, train_Y, dataframe_test_1, dataframe_test_modeloutput)

main()