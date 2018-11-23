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

PATH = "C:\\Users\\bitani\\Desktop\\INF8215\\TP3\\data\\"
X_train = pd.read_csv(PATH + "train.csv")
X_test = pd.read_csv(PATH + "test.csv")

X_train = X_train.drop(columns = ["OutcomeSubtype","AnimalID"])
X_test = X_test.drop(columns = ["ID"])

X_train, y_train = X_train.drop(columns = ["OutcomeType"]),X_train["OutcomeType"]

X_test.head()

y_train.head()

X_train1 = pd.read_csv("C:\\Users\\bitani\\Desktop\\INF8215\\TP3\\data\\train_preprocessed.csv")
X_test1 = pd.read_csv("C:\\Users\\bitani\\Desktop\\INF8215\\TP3\\data\\test_preprocessed.csv")

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

full_pipeline = ColumnTransformer(
    [
        ("bilal", pipeline_ageuponoutcome_changeToWeeks, "AgeuponOutcome"),
        ("Xiangyi", pipeline_AnimalType_ChangeAnimalType, "AnimalType"),
        ("Mohammed", pipeline_SexuponOutcome_ChangeSexUponOutcome, "SexuponOutcome")
    ], remainder='passthrough'
)

columns = ["AgeuponOutcome", "AnimalType", "Neutered Male", "Spayed Female", "Intact Male", "Intact Female", "Unknown", "Breed"]
#columns = ["AgeuponOutcome"]
X_train = pd.DataFrame(full_pipeline.fit_transform(X_train), columns= columns)


#t = X_train["SexuponOutcome"].value_counts()/len(X_train)

#print(t)

#print(convertAgeUponOutcomeToWeeks("10 months"))