from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):  
    """A softmax classifier"""

    def __init__(self, lr = 0.1, alpha = 100, n_epochs = 1000, eps = 1.0e-5,threshold = 1.0e-10 , regularization = True, early_stopping = True):
       
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

        

        tmp=np.ones((X.shape[0],1))
        X_bias = np.concatenate((tmp,X),axis=1)
        self.theta_=np.random.rand(self.nb_feature+1,self.nb_classes)

        

        for epoch in range( self.n_epochs):

            #logits =
            probabilities =self.predict_proba(X)
            
            
            loss =  self._cost_function(probabilities,y)
            gradient=self._get_gradient(X,y,probabilities)
            self.theta_ = self.theta_-self.lr*gradient
            
            self.losses_.append(loss)

            if self.early_stopping:
                if epoch>1 and abs(self.losses_[epoch]-self.losses_[epoch-1])<self.threshold:
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
        tmp=np.ones((X.shape[0],1))
        X_bias = np.concatenate((tmp,X),axis=1)
        z=np.dot(X_bias,self.theta_)
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
        predicted_prob=self.predict_proba(X,None)
        result=np.argmax(predicted_prob,axis=1)+1
        return result.reshape(result.shape[0],1)



    

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X,y)


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
        sum_number=X.shape[0]
        result=self.predict(X,y)
        tmp=result-y
        right_number=sum(tmp==0)
        return right_number[0]/sum_number


    

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
    
    def _cost_function(self,probabilities, y ):
        log_prob=np.exp(probabilities)
        log_prob[np.where(log_prob<self.eps)]=self.eps
        log_prob[np.where(log_prob>1-self.eps)]=1-self.eps
        cost_sum=0
        if self.regularization:
            for i in range(y.shape[0]):
                tmp_idx=int(y[i][0])
                cost_sum+=log_prob[i][tmp_idx]
                cost_sum=(-1/y.shape[0])*cost_sum
                cost_sum+=self.alpha*np.sum(np.power(self.theta_,2))
        else:
            for i in range(y.shape[0]):
                tmp_idx=int(y[i][0])
                cost_sum+=log_prob[tmp_idx][i]
                cost_sum=(-1/y.shape[0])*cost_sum
        return cost_sum

    

    
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

    
    
    def _one_hot(self,y):
        tmp=np.zeros((y.shape[0],self.nb_classes))
        for i in range(y.shape[0]):
            invoked_classc=int(y[i][0])
            tmp[i][invoked_classc]=1
        return tmp
    """
        In :
        Logits: (self.nb_features +1) * self.nb_classes

        Do:
        Compute softmax on logits

        Out:
        Probabilities
    """
    
    def _softmax(self,z):
        prob_vector=np.exp(z)
        tmp_sum=np.sum(prob_vector,axis=1)
        tmp_sum=tmp_sum.reshape(tmp_sum.shape[0],1)
        return prob_vector/tmp_sum


    

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

    def _get_gradient(self,X,y, probas):
        if self.regularization:
            m=y.shape[0]
            tmp=np.ones((X.shape[0],1))
            X_bias = np.concatenate((tmp,X),axis=1)
            tmp_subtraction=probas-self._one_hot(y)
            gradients_costfunction=(1/m)*np.dot((np.transpose(X_bias)),tmp_subtraction)+2*self.alpha*self.theta_
        else:
            m=y.shape[0]
            tmp=np.ones((X.shape[0],1))
            X_bias = np.concatenate((tmp,X),axis=1)
            gradients_costfunction=(1/m)*np.dot((np.transpose(X_bias)),probas-self._one_hot(y))
        return gradients_costfunction

    



