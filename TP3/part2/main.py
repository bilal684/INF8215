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
import csv


import collections,re


def readGroupCsv():
    dict = {}
    numbers = {}
    with open('group.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        idx = 0
        for row in csv_reader:
            dict[row[0].lower().strip()] = row[1].lower().strip()
            if row[1].lower().strip() not in numbers:
                numbers[row[1].lower().strip()] = idx
                idx = idx + 1
    return dict, numbers

dictCategory, dictValue = readGroupCsv()

PATH="C:\\Users\\bitani\\Desktop\\INF8215\\TP3\\data\\"
X_train = pd.read_csv(PATH+"train.csv")
X_test = pd.read_csv(PATH+"test.csv")

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


#dict = readGroupCsv()

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
    for key, value in dictCategory.items():
        if key in text:
            return dictValue[value]
    return 13

    #textWords = text.split(" ")
    constructedWord = ""
    #for i in len(textWords):
    #    constructedWord += textWords[i]
    #    if constructedWord in dict:
    #        return dict[constructedWord]
    #    else:
    #        constructedWord += " "
    #return "Unknown"

#def convertBreed(text):
    # if "mix" in text or "/" in text:
    #    return 1
    # else:
    #    return 0


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
        ('mohammed5', TransformationWrapper(transformation=convertBreed)),
        ('encode', OneHotEncoder(categories='auto', sparse=False))

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

columns = ["AgeuponOutcome", "AnimalType", "Neutered Male", "Spayed Female", "Intact Male", "Intact Female", "UnknownSex", "toy",
           "hound", "terrier", "working", "non-sporting", "herding", "sporting", "terrier & toy", "non-sporting & toy",
           "pit bull", "longhair", "shorthair", "mediumhair", "Unknown"]
X_train = pd.DataFrame(full_pipeline.fit_transform(X_train), columns=columns)
X_train.to_csv("x_train", sep=',', encoding='utf-8')
X_test = pd.DataFrame(full_pipeline.transform(X_test), columns= columns)

X_train_all = pd.concat([X_train, X_train1], axis=1)
X_test_all = pd.concat([X_test, X_test1], axis=1)



target_label = LabelEncoder()
y_train_label = target_label.fit_transform(y_train)

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):
    """A softmax classifier"""

    def __init__(self, lr=0.1, alpha=1, n_epochs=1000, eps=1.0e-5, threshold=1.0e-5, regularization=True,
                 early_stopping=True):

        """
            self.lr : the learning rate for weights update during gradient descent
            self.alpha: the regularization coefficient
            self.n_epochs: the number of iterations
            self.eps: the threshold to keep probabilities in range [self.eps;1.-self.eps]
            self.regularization: Enables the regularization, help to prevent overfitting
            self.threshold: Used for early stopping, if the difference between losses during
                            two consecutive epochs is lower than self.threshold, then we stop the algorithm
            self.early_stopping: enables early stopping to prevent overfitting
        """

        self.lr = lr
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.eps = eps
        self.regularization = regularization
        self.threshold = threshold
        self.early_stopping = early_stopping

    """
        Public methods, can be called by the user
        To create a custom estimator in sklearn, we need to define the following methods:
        * fit
        * predict
        * predict_proba
        * fit_predict        
        * score
    """

    """
        In:
        X : the set of examples of shape nb_example * self.nb_features
        y: the target classes of shape nb_example *  1

        Do:
        Initialize model parameters: self.theta_
        Create X_bias i.e. add a column of 1. to X , for the bias term
        For each epoch
            compute the probabilities
            compute the loss
            compute the gradient
            update the weights
            store the loss
        Test for early stopping

        Out:
        self, in sklearn the fit method returns the object itself


    """

    def fit(self, X, y=None):

        prev_loss = np.inf
        self.losses_ = []

        self.nb_feature = X.shape[1]
        self.nb_classes = len(np.unique(y))
        tmp = np.ones((X.shape[0], 1))
        X_bias = np.concatenate((tmp, X), axis=1)
        self.theta_ = np.random.rand(self.nb_feature + 1, self.nb_classes)
        for epoch in range(self.n_epochs):
            probabilities = self.predict_proba(X)
            loss = self._cost_function(probabilities, y)
            gradient = self._get_gradient(X, y, probabilities)
            self.theta_ = self.theta_ - self.lr * gradient
            self.losses_.append(loss)
            if self.early_stopping:
                if epoch > 1 and abs(self.losses_[epoch] - self.losses_[epoch - 1]) < self.threshold:
                    break
        return self

    """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax

        Out:
        Predicted probabilities
    """

    def predict_proba(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        tmp = np.ones((X.shape[0], 1))
        X_bias = np.concatenate((tmp, X), axis=1)
        z = np.dot(X_bias, self.theta_)
        return self._softmax(z)
        """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax
        Predict the classes

        Out:
        Predicted classes
    """

    def predict(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        predicted_prob = self.predict_proba(X, None)
        result = np.argmax(predicted_prob, axis=1)
        return result.reshape(result.shape[0], 1)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X, y)

    """
        In : 
        X set of examples (without bias term)
        y the true labels

        Do:
            predict probabilities for X
            Compute the log loss without the regularization term

        Out:
        log loss between prediction and true labels

    """

    def score(self, X, y=None):
        sum_number = X.shape[0]
        result = self.predict(X, y)
        tmp = result - y
        right_number = sum(tmp == 0)
        return right_number[0] / sum_number

    """
        Private methods, their names begin with an underscore
    """

    """
        In :
        y without one hot encoding
        probabilities computed with softmax

        Do:
        One-hot encode y
        Ensure that probabilities are not equal to either 0. or 1. using self.eps
        Compute log_loss
        If self.regularization, compute l2 regularization term
        Ensure that probabilities are not equal to either 0. or 1. using self.eps

        Out:
        Probabilities
    """

    def _cost_function(self, probabilities, y):
        m = probabilities.shape[0]
        p_processed = np.clip(probabilities, self.eps, 1 - self.eps)
        cost_sum = 0
        y_one_hot = self._one_hot(y)
        for i in range(m):
            c = np.argmax(y_one_hot[i, :])  # class
            cost_sum += np.log(p_processed[i, c])
        if self.regularization:
            cost_sum = (-1 / m) * cost_sum
            cost_sum += (1 / m) * self.alpha * np.sum(self.theta_[1:, :] ** 2)
            return cost_sum
        else:
            return (-1 / m) * cost_sum

    """
        In :
        Target y: nb_examples * 1

        Do:
        One hot-encode y
        [1,1,2,3,1] --> [[1,0,0],
                         [1,0,0],
                         [0,1,0],
                         [0,0,1],
                         [1,0,0]]
        Out:
        y one-hot encoded
    """

    def _one_hot(self, y):
        tmp = np.zeros((y.shape[0], self.nb_classes))
        for i in range(y.shape[0]):
            invoked_classc = int(y[i][0])
            tmp[i][invoked_classc] = 1
        return tmp

    """
        In :
        Logits: (self.nb_features +1) * self.nb_classes

        Do:
        Compute softmax on logits

        Out:
        Probabilities
    """

    def _softmax(self, z):
        prob_vector = np.exp(z)
        tmp_sum = np.sum(prob_vector, axis=1)
        tmp_sum = tmp_sum.reshape(tmp_sum.shape[0], 1)
        return prob_vector / tmp_sum

    """
        In:
        X with bias
        y without one hot encoding
        probabilities resulting of the softmax step

        Do:
        One-hot encode y
        Compute gradients
        If self.regularization add l2 regularization term

        Out:
        Gradient

    """

    def _get_gradient(self, X, y, probas):
        if self.regularization:
            m = y.shape[0]
            tmp = np.ones((X.shape[0], 1))
            X_bias = np.concatenate((tmp, X), axis=1)
            tmp_labels = self._one_hot(y)
            tmp_subtraction = probas - tmp_labels
            gradients_costfunction = (1 / m) * (np.dot((np.transpose(X_bias)), tmp_subtraction))
            gradients_costfunction[1:, :] += 2 * self.alpha * self.theta_[1:, :] / m
        else:
            m = y.shape[0]
            tmp = np.ones((X.shape[0], 1))
            X_bias = np.concatenate((tmp, X), axis=1)
            gradients_costfunction = (1 / m) * np.dot((np.transpose(X_bias)), probas - self._one_hot(y))
        return gradients_costfunction


selected_model = SoftmaxClassifier()
y_train_label=y_train_label.reshape((y_train_label.shape[0],1))
y_pred = selected_model.fit_predict(X_train,y_train_label)

print("Hello")


# cross-validation
#from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
#from Draw_Charts import Draw_Charts
#from SoftmaxClassifier import SoftmaxClassifier
#from sklearn.metrics import precision_recall_fscore_support
#import matplotlib.pyplot as plt
#Draw_Charts(y_train_label,"Original Ratio")
#sfolder = StratifiedKFold(n_splits=10,random_state=0,shuffle=False)
#y_train_label=y_train_label.reshape((y_train_label.shape[0],1))
#X_train_all=X_train_all.values
#scores_list=[]
##random sampling
#X_train, X_test, y_train, y_test = train_test_split( X_train_all, y_train_label, test_size=0.1, random_state=42)
##Draw_Charts(y_test,"Random Sampling")
#for train_index, test_index in sfolder.split(X_train_all, y_train_label):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_cross_train=X_train_all[train_index]
#    y_cross_train=y_train_label[train_index]
#    X_cross_test=X_train_all[test_index]
#    y_cross_test=y_train_label[test_index]
##    Draw_Charts(y_cross_test,"StratifiedKFold")
#    cl = SoftmaxClassifier()
##    train_p=cl.fit_predict(X_cross_train,y_cross_train)
 #   scores = cl.score(X_cross_test,y_cross_test)
#    scores_list.append(scores)
#    print (scores)
  #  print("train : "+ str(precision_recall_fscore_support(y_cross_train, train_p,average = "macro")))
   # print("test : "+ str(precision_recall_fscore_support(y_cross_test, test_p,average = "macro")))
#    plt.plot(cl.losses_)
#    plt.show()
#    print("hello")
#plt.plot(scores_list)








# display precision, recall and f1-score on train/test set





