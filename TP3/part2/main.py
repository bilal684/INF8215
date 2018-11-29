import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from preprocessing import TransformationWrapper
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

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



target_label = LabelEncoder()
y_train_label = target_label.fit_transform(y_train)


# cross-validation
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from Draw_Charts import Draw_Charts
from SoftmaxClassifier import SoftmaxClassifier
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
Draw_Charts(y_train_label,"Original Ratio")
sfolder = StratifiedKFold(n_splits=10,random_state=0,shuffle=False)
y_train_label=y_train_label.reshape((y_train_label.shape[0],1))
X_train_all=X_train_all.values
scores_list=[]
#random sampling
X_train, X_test, y_train, y_test = train_test_split( X_train_all, y_train_label, test_size=0.1, random_state=42)
Draw_Charts(y_test,"Random Sampling")
for train_index, test_index in sfolder.split(X_train_all, y_train_label):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_cross_train=X_train_all[train_index]
    y_cross_train=y_train_label[train_index]
    X_cross_test=X_train_all[test_index]
    y_cross_test=y_train_label[test_index]
    Draw_Charts(y_cross_test,"StratifiedKFold")
    cl = SoftmaxClassifier()
    train_p=cl.fit_predict(X_cross_train,y_cross_train)
    scores = cl.score(X_cross_test,y_cross_test)
    scores_list.append(scores)
    print (scores)
  #  print("train : "+ str(precision_recall_fscore_support(y_cross_train, train_p,average = "macro")))
   # print("test : "+ str(precision_recall_fscore_support(y_cross_test, test_p,average = "macro")))
    plt.plot(cl.losses_)
    plt.show()
    print("hello")
plt.plot(scores_list)








# display precision, recall and f1-score on train/test set





