import numpy as np 
import pandas as pd 


class LogisticRegression:
    
    def __init__(self):
        # Added a 3rd theta to compensate for the bias
        self.theta = np.array([1, 1, 0])

    def fit(self, X_df, y_df):
        """
        Estimates parameters for the classifier
        
        Args:
            X_df (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y_df (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        X, y = np.asarray(X_df).T, np.asarray(y_df)
        
        # Added ones is to make X homogenous so the bias can be
        # computed with the 3rd theta param
        X = np.concatenate([X, np.ones((1,len(y)))])
        learning_rate = 1
        
        for epoch in range(100):
            error = y - self.h(X)
            gradient = X @ error
            self.theta = self.theta + learning_rate * gradient/len(y)
      
    def h(self, x):
        return (1 / (1 + np.exp(-self.theta @ x)))
    
    def predict(self, X_df):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X_df (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        X = np.asarray(X_df).T
        X = np.concatenate([X, np.ones((1,X.shape[1]))])
        return self.h(X)
    
        
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
