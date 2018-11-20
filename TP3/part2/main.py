import pandas as pd
from sklearn.impute import SimpleImputer

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

def convertAgeUponTimeToDays(text):
    number, timeFrame = text.split(" ")
    if timeFrame.lower() == "year" or timeFrame.lower() == "years":
        return float(number) * 52.0 #52 weeks per year
    elif timeFrame.lower() == "month" or timeFrame.lower() == "months":
        return float(number) * 4.0 #4 weeks per month
    elif timeFrame.lower() == "week" or timeFrame.lower() == "weeks":
        return float(number) # 1 week per week
    elif timeFrame.lower() == "day" or timeFrame.lower() == "days":
        return float(number) * 1.0/7.0 #because there are 7 days per week

X_train["AgeuponOutcome"].value_counts()/len(X_train)



#print(convertAgeUponTimeToDays("10 day"))