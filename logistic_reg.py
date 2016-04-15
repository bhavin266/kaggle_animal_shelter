import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

train_X = []
train_Y = []

def generate_train_X(dataframe):

    for row in dataframe.iterrows():
        list(row[-1])
        train_X.append(list(row[-1]))

def generate_train_Y(dataframe):

    for row in dataframe.iterrows():
        list(row[-1])
        train_Y.extend(list(row[-1]))

def perform_logistic_regression(train_X, train_Y, dataframe_test, dataframe_test_modeloutput):

    predict_list = []
    X = np.array(train_X).astype(float)
    Y = np.array(train_Y).astype(float)
    regression = LogisticRegression()
    regression.fit(X, Y)
    print(regression.score(dataframe_test, dataframe_test_modeloutput)*100)

def main():
    dataframe = pd.read_csv('training_animals_normalized_vectorized.csv')
    # dataframe1 = dataframe[['AnimalType','SexuponOutcome','AgeuponOutcome','Breed','Color']]
    li1 = ["Mix","AnimalType=Cat","AnimalType=Dog","BreedGroup=Herding","BreedGroup=Hound","BreedGroup=Non-Sporting","BreedGroup=Sporting","BreedGroup=Terrier","BreedGroup=Toy","BreedGroup=Unknown","BreedGroup=Working","Quality=Intact","Quality=Neutered","Quality=Spayed","Quality=Unknown","Sex=Female","Sex=Male","Sex=Unknown"]
    dataframe1 = dataframe[li1]
    dataframe1= dataframe1.fillna(-1)
    generate_train_X(dataframe1)
    dataframe2 = dataframe[['OutcomeType']]
    generate_train_Y(dataframe2)
    dataframe_test = pd.read_csv('testing_animals_normalized_vectorized.csv')
    dataframe_test_1 = dataframe_test[li1]
    dataframe_test_modeloutput = dataframe_test[['OutcomeType']]
    dataframe_test_modeloutput = dataframe_test_modeloutput.fillna(-1)
    dataframe_test_1 = dataframe_test_1.fillna(-1)
    perform_logistic_regression(train_X, train_Y, dataframe_test_1, dataframe_test_modeloutput)


main()
