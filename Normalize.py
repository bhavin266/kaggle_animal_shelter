import re
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.feature_extraction import DictVectorizer

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

def get_categories(data):
    li_outcomes = data['OutcomeType'].drop_duplicates().tolist()
    li_animal_type = data['AnimalType'].drop_duplicates().tolist()
    #li_sex_type = data['SexuponOutcome'].drop_duplicates().tolist()
    li_sex = data['Sex'].drop_duplicates().tolist()
    li_quality = data['Quality'].drop_duplicates().tolist()
    li_breed_type = data['Breed'].drop_duplicates().tolist()
    li_color_type = data['Color'].drop_duplicates().tolist()
    li_name = data['Name'].drop_duplicates().tolist()
    li_LifeStage = data['LifeStage'].drop_duplicates().tolist()
    li_DayPeriod = data['DayPeriod'].drop_duplicates().tolist()
    li_HairType = data["HairType"].drop_duplicates().tolist()
    li_BreedGroup = data["BreedGroup"].drop_duplicates().tolist()
    return li_animal_type, li_breed_type, li_outcomes, li_color_type, li_sex,li_quality,li_name,li_LifeStage ,li_DayPeriod,li_HairType,li_BreedGroup 

def one_hot_dataframe(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.
    """
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)

def getNormalizedData():
    data = pd.read_csv("train_animal.csv")
    breedDict = {}
    li = []
    with open('BreedGroups.csv') as f:
        li = f.readlines()
    for el in li:
        arr = el.strip().split(',')
        breedDict[arr[0]] = arr[1]

    data = data[['AnimalID','Name','DateTime','OutcomeType','AnimalType','SexuponOutcome','AgeuponOutcome','Breed','Color']]
    
    data['Name'].fillna('Unknown',inplace=True)
    data.loc[~data.Name.str.contains('Unknown'),'Name'] = 'Known'

    data = data.dropna()
    data.loc[data.SexuponOutcome == "Unknown","SexuponOutcome"] = "Unknown Unknown"
    data["Quality"],data["Sex"] = zip(*data["SexuponOutcome"].str.split().tolist())
    del data['SexuponOutcome']

    li = ['Shorthair','Longhair','Medium Hair','Wirehair','Rough','Smooth Coat','Smooth','Black/Tan','Flat Coat','Coat']
    data['matches'] = data.Breed.apply(lambda sentence: [word for word in li if word in sentence])
    mask = data.matches.apply(len) > 0
    data["HairType"] = "Unknown"
    data.loc[mask, 'HairType'] = data.loc[mask, 'matches'].str[0]
    data.loc[mask, 'Breed'] = (data.loc[mask, 'Breed'].apply(lambda sentence: 
                                                       reduce(lambda remaining_sentence, word: 
                                                           remaining_sentence.replace(word, ''), li, sentence)))
    del data['matches']

    data['Mix'] = 0
    data.loc[data.Breed.str.contains('Mix'),'Mix'] = 1
    data.loc[data.Breed.str.contains('/'),'Mix'] = 1
    data['Breed'] = data['Breed'].map(lambda x: x.rstrip('Mix').strip())
    data['Breed'] = data['Breed'].map(lambda x: x.split('/')[0].strip())

    data["BreedGroup"] = "Unknown"
    data.BreedGroup = data.Breed.map(breedDict).fillna(data.BreedGroup)

    data['MultiColored'] = 0
    data.loc[data.Breed.str.contains('/'),'MultiColored'] = 1
    data['Color'] = data['Color'].map(lambda x: x.split('/')[0].strip())

    data['Date'],data['Time'] = zip(*data["DateTime"].str.split().tolist())
    #for el in data["Time"].str.split(':'):
    #    print(type(el))
    #    print(el)
    data["Hour"],data["Minute"],data["Second"] = zip(*[[int(el[0]),int(el[1]),int(el[2])] for el in data["Time"].str.split(':').tolist()])
    data["Year"],data["Month"],data["Day"] = zip(*data["Date"].str.split('-').tolist())
    del data['DateTime']
    del data['Date']
    del data['Time']

    days = lambda x : toDays(x)
    data['AgeuponOutcome'] = data['AgeuponOutcome'].apply(days)
    
    data["LifeStage"] = ""
    data.loc[data.AgeuponOutcome <= 365,"LifeStage"] = "Young"
    data.loc[data.AgeuponOutcome > 365,"LifeStage"] = "Old"

    data["DayPeriod"] = ""
    data.loc[(data.Hour > 5) & (data.Hour < 13),"DayPeriod"] = "Morning"
    data.loc[(data.Hour > 12) & (data.Hour < 17),"DayPeriod"] = "AfterNoon"
    data.loc[(data.Hour > 16) & (data.Hour < 20),"DayPeriod"] = "Morning"
    data.loc[(data.Hour > 19) | (data.Hour < 6),"DayPeriod"] = "Night"

    li_animal_type, li_breed_type, li_outcomes, li_color_type, li_sex,li_quality,li_name,li_LifeStage,li_DayPeriod,li_HairType,li_BreedGroup = get_categories(data)

    data1 = data.copy()
    data1=data1[["OutcomeType","Quality","Sex","BreedGroup","AnimalType","Mix"]]
    for i in range(len(li_outcomes)):
        type_ = li_outcomes[i]
        data1.OutcomeType[data1.OutcomeType == type_] = i
    df,_,_ = one_hot_dataframe(data1,["Quality","Sex","BreedGroup","AnimalType"],True)
    df = df.sample(frac=1)
    trainingLimit = 9 * len(df) // 10
    trainingData = df[:trainingLimit]
    testingData = df[trainingLimit:]
    trainingData.to_csv("training_animals_normalized_vectorized.csv",sep=',', encoding='utf-8',index=False)
    testingData.to_csv("testing_animals_normalized_vectorized.csv",sep=',', encoding='utf-8',index=False)
    
    for i in range(len(li_animal_type)):
        type_ = li_animal_type[i]
        data.AnimalType[data.AnimalType == type_] = i

    for i in range(len(li_outcomes)):
        type_ = li_outcomes[i]
        data.OutcomeType[data.OutcomeType == type_] = i

    for i in range(len(li_sex)):
        type_ = li_sex[i]
        data.Sex[data.Sex == type_] = i

    for i in range(len(li_quality)):
        type_ = li_quality[i]
        data.Quality[data.Quality == type_] = i

    for i in range(len(li_color_type)):
        type_ = li_color_type[i]
        data.Color[data.Color == type_] = i

    for i in range(len(li_breed_type)):
        type_ = li_breed_type[i]
        data.Breed[data.Breed == type_] = i

    for i in range(len(li_name)):
        type_ = li_name[i]
        data.Name[data.Name == type_] = i

    for i in range(len(li_LifeStage)):
        type_ = li_LifeStage[i]
        data.LifeStage[data.LifeStage == type_] = i

    for i in range(len(li_DayPeriod)):
        type_ = li_DayPeriod[i]
        data.DayPeriod[data.DayPeriod == type_] = i
    
    for i in range(len(li_HairType)):
        type_ = li_HairType[i]
        data.HairType[data.HairType == type_] = i

    for i in range(len(li_BreedGroup)):
        type_ = li_BreedGroup[i]
        data.BreedGroup[data.BreedGroup == type_] = i

    #data.apply(np.random.shuffle,axis=0)
    data = data.sample(frac=1)
    trainingLimit = 9 * len(data) // 10
    trainingData = data[:trainingLimit]
    testingData = data[trainingLimit:]
    return trainingData,testingData, li_outcomes,li_animal_type,li_sex,li_quality,li_breed_type,li_color_type,li_name,li_LifeStage ,li_DayPeriod,li_HairType,li_BreedGroup

trainingData,testingData, li_outcomes,li_animal_type,li_sex,li_quality,li_breed_type,li_color_type,li_name,li_LifeStage,li_DayPeriod,li_HairType,li_BreedGroup = getNormalizedData()
trainingData.to_csv("training_animals_normalized.csv",sep=',', encoding='utf-8',index=False)
testingData.to_csv("testing_animals_normalized.csv",sep=',', encoding='utf-8',index=False)

#print(trainingData)
def WriteRefData(name,li,myfile):
    myfile.write(name + '\n')
    for i in range(len(li)):
        if type(li[i]) != float:
            myfile.write(str(i) + ':' + li[i] + '\n')
        else:
            myfile.write(str(i) + ':NaN\n')
    myfile.write('***************\n')

with open("refData.txt", "w") as myfile:
    WriteRefData("OutcomeType",li_outcomes,myfile)
    WriteRefData("AnimalType",li_animal_type,myfile)
    WriteRefData("SexType",li_sex,myfile)
    WriteRefData("Quality",li_quality,myfile)
    WriteRefData("Name",li_name,myfile)
    WriteRefData("LifeStage",li_LifeStage,myfile)
    WriteRefData("DayPeriod",li_DayPeriod ,myfile)
    WriteRefData("HairType",li_HairType,myfile)
    WriteRefData("BreedGroup",li_BreedGroup,myfile)
    
    WriteRefData("ColorType",li_color_type,myfile)
    

