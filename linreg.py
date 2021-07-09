import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def normalize(X): # creating standard variables here (u-x)/sigma
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return
    for j in range(X.shape[1]):
        u = np.mean(X[:,j])
        s = np.std(X[:,j])
        X[:,j] = (X[:,j] - u) / s

def MSE(X, y, B, lmbda):
    """
    X.shape: (n, p+1)
    B.shape: (p+1, 1)
    y.shape: (n, 1)
    return a 1 by 1 scalar
    """
    return np.dot(np.transpose(y - np.dot(X, B)), y - np.dot(X, B))

def loss_gradient(X, y, B, lmbda):
    """
    X.shape: (n, p+1)   add the column of one's to X
    B.shape: (p+1, 1)   put beta_0 into beta
    y.shape: (n, 1)
    return a vector of shape (p+1, 1)
    """
    return - np.dot(np.transpose(X), y - np.dot(X, B))

def loss_ridge(X, y, B, lmbda):
    """
    X.shape: (n, p)  don't add the column of one's to X
    B.shape: (p, 1)  don't put beta_0 into beta
    y.shape: (n, 1)
    lmbda: scalar
    return a scalar
    """
    return np.dot(np.transpose(y - np.dot(X, B)), y - np.dot(X, B)) + lmbda * np.dot(np.transpose(B), B)

def loss_gradient_ridge(X, y, B, lmbda):
    """
    X.shape: (n, p)   don't add the column of one's to X
    B.shape: (p, 1)   don't put beta_0 into beta
    y.shape: (n, 1)
    return a vector of shape (p, 1)
    """
    return - np.dot(np.transpose(X), y - np.dot(X, B)) + lmbda * B

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, B,lmbda):
    """
    X.shape: (n, p+1)   add the column of one's to X
    B.shape: (p+1, 1)   put beta_0 into beta
    y.shape: (n, 1)
    return a scalar
    """
    # return np.sum(y * np.dot(X, B) - np.log(1 + np.exp(np.dot(X, B))))
    return np.sum(np.multiply(y, np.dot(X, B)) - np.log(1 + np.exp(np.dot(X, B))))

def log_likelihood_gradient(X, y, B, lmbda):
    """
    X.shape: (n, p+1)   add the column of one's to X
    B.shape: (p+1, 1)   put beta_0 into beta
    y.shape: (n, 1)
    return a vector of shape (p+1, 1)
    """
    return -np.dot(np.transpose(X), y-sigmoid(np.dot(X, B)))

# NOT REQUIRED but to try to implement for fun
def L1_log_likelihood(X, y, B, lmbda):
    pass

# NOT REQUIRED but to try to implement for fun
def L1_log_likelihood_gradient(X, y, B, lmbda):
    """
    Must compute \beta_0 differently from \beta_i for i=1..p.
    \beta_0 is just the usual log-likelihood gradient
    # See https://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html
    # See https://stackoverflow.com/questions/38853370/matlab-regularized-logistic-regression-how-to-compute-gradient
    """
    pass

def minimize(X, y, loss, loss_gradient,
              eta=0.00001, lmbda=0.0,
              max_iter=1000, addB0=True,
              precision=1e-9):
    "Here are various bits and pieces you might want"
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")
    n, p = X.shape
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")
    
    if addB0:  # add column of 1s to X, add B0 to B 
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        B = np.random.random_sample(size=(p+1, 1)) * 2 - 1  # make between [-1,1) 
        h = np.zeros(shape=(p+1, 1))   # p+1 sized sum of squared gradient history 
    else:     
        B = np.random.random_sample(size=(p, 1)) * 2 - 1  # make between [-1,1) 
        h = np.zeros(shape=(p, 1))     # p+1 sized sum of squared gradient history 

    step = 0
    eps = 1e-5 # prevent division by 0
    
    while step < max_iter:
        cur_loss_gradient = loss_gradient(X, y, B, lmbda)
        if np.linalg.norm(cur_loss_gradient) >= precision:
            step += 1
            h += np.square(cur_loss_gradient)  # track sum of squared partials, use element-wise product
            B = B - eta * cur_loss_gradient / (np.sqrt(h) + eps)
        else:
            break
            
    return B


class LinearRegression621: # REQUIRED
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                           MSE,
                           loss_gradient,
                           self.eta,
                           self.lmbda,
                           self.max_iter)


class LogisticRegression621: # REQUIRED
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter
        
    def predict_proba(self, X):
        """
        Compute the probability that the target is 1. Basically do
        the usual linear regression and then pass through a sigmoid.
        """
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return sigmoid(np.dot(X, self.B))
        

    def predict(self, X):
        """
        Call self.predict_proba() to get probabilities then, for each x in X,
        return a 1 if P(y==1,x) > 0.5 else 0.
        """
        probs = self.predict_proba(X)
        prob2target = lambda p: 1 if p > 0.5 else 0
        vfunc = np.vectorize(prob2target) 
        return vfunc(probs)
    
    def fit(self, X, y):
        self.B = minimize(X, y, 
                          loss = log_likelihood,   # loss function
                          loss_gradient = log_likelihood_gradient,   # loss gradient funtion
                          addB0 = True, 
                          eta = self.eta,
                          lmbda = self.lmbda,
                          max_iter = self.max_iter)


class RidgeRegression621: # REQUIRED
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter
        
    def fit(self, X, y):
        B_1_to_p = minimize(X, y,
                            loss = loss_ridge,           # loss function
                            loss_gradient = loss_gradient_ridge,  # loss gradient funtion
                            addB0 = False,
                            eta = self.eta,
                            lmbda = self.lmbda,
                            max_iter = self.max_iter)
        B_0 = np.array([y.mean()])
        self.B = np.concatenate([B_0, B_1_to_p.reshape(-1)])

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

# NOT REQUIRED but to try to implement for fun
class LassoLogistic621:
    pass
