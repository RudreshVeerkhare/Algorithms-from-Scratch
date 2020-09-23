import numpy as np
class NaiveBayes:
    def __init__(self):
        self.masks = None # [class_mask, features]
        self.classes = None
        self.means = None # [classes, features]
        self.sigmas = None # [classes, features]
        self.class_probs = None

    
    def fit(self, X, Y):
        self.masks = self.getMasks(Y)
        self.means = self.getMeans(X)
        self.sigmas = self.getSigmas(X)
        self.class_probs = self.masks.mean(axis=0)
        return self
    
    def getMeans(self, X):
        means = list()
        for _class in self.masks.T:
            means.append(X[_class].mean(axis=0))
        return np.array(means)
    
    def getSigmas(self, X):
        sigmas = list()
        for _class in self.masks.T:
            sigmas.append(X[_class].std(axis=0))
        return np.array(sigmas)
    
    @staticmethod
    def likelyhood(x, mean, sigma):
        """normal distribution"""
        return np.exp(-(x-mean)**2/(2*sigma**2))*(1/(np.sqrt(2*np.pi)*sigma))
        
    
    def getMasks(self, Y):
        temp = list()
        self.classes = np.unique(Y)
        for i in self.classes:
            temp.append(Y == i)
        return np.array(temp).T
        
    
    def predict(self, X, print_joint_probs = False):
        preds = list()
        for i in range(len(self.classes)):
            preds.append(np.prod(self.likelyhood(X, self.means[i, :], self.sigmas[i, :]), axis=1))
        preds = np.array(preds).T
        if print_joint_probs: print(preds)
        return self.classes[preds.argmax(axis=1)]