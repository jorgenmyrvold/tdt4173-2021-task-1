import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.theta = np.array([[1,1]])
        
        
    def fit(self, X_panadas, y_pandas):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        # TODO: Implement
        X, y = np.asarray(X_panadas).T, np.asarray(y_pandas)
        learning_rate = 0.01
        last_change = 99999
        
        print(np.sum(np.array([1,2]) - np.array([2,1])))

        while(last_change > 0.001):
            prev_theta = self.theta
            self.theta = self.theta + learning_rate * (np.sum(y) - np.sum(self.h(X)) * np.sum(X, axis=1))
            last_change = np.sum(prev_theta - self.theta)
      
    def h(self, x):
        return 1 / (1 + np.e ** (-self.theta @ x))

    
    def predict(self, X_pandas):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        # TODO: Implement
        X = np.asarray(X_pandas)
        return (1 / (1 + np.e ** (-self.theta @ X.T))).reshape(len(X))
    
        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

def logit(x):
    """
    Transforms to log
    """
    return np.log(x/(1-x))

def inverse_logit(x):
    """
    Transforms back to not-log
    """
    return (np.e ** np.log(x)) / (1 + (np.e ** np.log(x)))