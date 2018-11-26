import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from preprocessing import TransformationWrapper
from sklearn.preprocessing import OneHotEncoder
from preprocessing import LabelEncoderP
import math

import collections,re

PATH="E:\\polytechnique\\courses\\FAI\\TP3_Code\\TP\\TP3\\data\\"
X_train = pd.read_csv(PATH+"train.csv")
X_test = pd.read_csv(PATH+"test.csv")

X_train = X_train.drop(columns = ["OutcomeSubtype","AnimalID"])
X_test = X_test.drop(columns = ["ID"])

X_train, y_train = X_train.drop(columns = ["OutcomeType"]),X_train["OutcomeType"]

X_test.head()

y_train.head()

X_train1 = pd.read_csv("E:\\polytechnique\\courses\\FAI\\TP3_Code\\TP\\TP3\\data\\train_preprocessed.csv")
X_test1 = pd.read_csv("E:\\polytechnique\\courses\\FAI\\TP3_Code\\TP\\TP3\\data\\test_preprocessed.csv")

X_train1.head()

X_train = X_train.drop(columns = ["Color","Name","DateTime"])
X_test = X_test.drop(columns = ["Color","Name","DateTime"])

def convertAgeUponOutcomeToWeeks(text):
    if (not isinstance(text, str)) and math.isnan(text):
       return 0.0
    else:
        number, timeFrame = text.split(" ")
        if timeFrame.lower() == "year" or timeFrame.lower() == "years":
            return float(number) * 52.1428228589286 #52 weeks per year
        elif timeFrame.lower() == "month" or timeFrame.lower() == "months":
            return float(number) * 4.34524 #4 weeks per month
        elif timeFrame.lower() == "week" or timeFrame.lower() == "weeks":
            return float(number) # 1 week per week
        elif timeFrame.lower() == "day" or timeFrame.lower() == "days":
            return float(number) * 1.0/7.0 #because there are 7 days per week

def convertAnimalType(text):
    if text == 'Dog':
        return 0
    else:
        return 1

def convertBreed(text):
    text = text.lower()
    if "mix" in text or "/" in text:
        return 1
    else:
        return 0

def convertSexUponOutcome(text):
    if text == "Neutered Male":
        return 0
    elif text == "Spayed Female":
        return 1
    elif text == "Intact Male":
        return 2
    elif text == "Intact Female":
        return 3
    else:
        return 4
pipeline_ageuponoutcome_changeToWeeks = Pipeline([
    ('mohammed', TransformationWrapper(transformation=convertAgeUponOutcomeToWeeks)),
    ('mohammed2', StandardScaler())
])

pipeline_AnimalType_ChangeAnimalType = Pipeline(
    [
        ('mohammed3', TransformationWrapper(transformation=convertAnimalType))
    ]
)

pipeline_SexuponOutcome_ChangeSexUponOutcome = Pipeline(
    [
        ('mohammed4', TransformationWrapper(transformation=convertSexUponOutcome)),
        ('encode', OneHotEncoder(categories='auto', sparse=False))
    ]
)

pipeline_changeBreed = Pipeline(
    [
        ('mohammed5', TransformationWrapper(transformation=convertBreed))
    ]
)

full_pipeline = ColumnTransformer(
    [
        ("bilal", pipeline_ageuponoutcome_changeToWeeks, "AgeuponOutcome"),
        ("Xiangyi", pipeline_AnimalType_ChangeAnimalType, "AnimalType"),
        ("Mohammed", pipeline_SexuponOutcome_ChangeSexUponOutcome, "SexuponOutcome"),
        ('breed', pipeline_changeBreed, "Breed")
    ], remainder='passthrough'
)

columns = ["AgeuponOutcome", "AnimalType", "Neutered Male", "Spayed Female", "Intact Male", "Intact Female", "Unknown", "Mix"]
#columns = ["AgeuponOutcome"]
X_train = pd.DataFrame(full_pipeline.fit_transform(X_train), columns= columns)
X_test = pd.DataFrame(full_pipeline.transform(X_test), columns= columns)

X_train_all = pd.concat([X_train, X_train1], axis=1)
X_test_all = pd.concat([X_test, X_test1], axis=1)
print("hello")

#X_train['Breed']

#b = collections.Counter([y for x in X_train["Breed"].values.flatten() for y in x.split()])

#t = X_train["Breed"].value_counts()/len(X_train)

#print(b)

#print(convertAgeUponOutcomeToWeeks("10 months"))