import numpy as np  
class KMeans:
    def __init__(self, n_clusters = 8, similarity="cosine"):
        # static vars
        self.n_clusters = n_clusters
        self.similarity = similarity
        
        # dynamic var
        self.centroids = None # [n_clusters, n_features] matrix of centroids 
        self.n_features = None # number of features
        self.dists = None # [N(no of points), n_clusters] matrix for dist of each point from each centroid
        self.labels = None # label of each point (i.e cluster to which point belongs)
        self.losses = [float('inf')] # list of all the losses 
        
    def fit(self, X: np.ndarray, n_iterations=10000):
        
        # init
        self.n_features = X.shape[1]
#         self.centroids = np.random.randn(self.n_clusters, self.n_features)
        center_id = np.random.randint(X.shape[0], size=self.n_clusters)
        self.centroids = X[center_id, :]
        print(self.centroids.shape)
        self.dists = np.zeros((X.shape[0], self.n_clusters))
        
        for _ in range(n_iterations):
            self.calculateDist(X)
            self.assignLabels(X)
            self.calculateLoss(X)
            self.getNewCentroids(X)
            if self.isConverged():
                break
        
        
        return self.labels
    
    def calculateDist(self, X: np.ndarray):
        for i in range(self.n_clusters):
            if self.similarity == "cosine":
                self.dists[:, i] = self.cosineNorm(X, self.centroids[i, :]).ravel()
            elif self.similarity == "euclidean":
                self.dists[:, i] = self.ecludianNorm(X, self.centroids[i, :]).ravel()
    
    def assignLabels(self, X: np.ndarray):
        self.labels = np.argmin(self.dists, axis=1).reshape(-1, 1)
    
    def calculateLoss(self, X:np.ndarray):
        loss = 0
        for i in range(self.n_clusters):
            mask = (self.labels == i).ravel()
            loss += np.sum(self.dists[mask, i])
        self.losses.append(loss)
    
    def getNewCentroids(self, X: np.ndarray):
        for i in range(self.n_clusters):
            mask = (self.labels == i).ravel()
            x = X[mask]
            self.centroids[i] = np.mean(x, axis=0)
    
    def isConverged(self):
        if self.losses[-1] >= self.losses[-2]:
            return True
        return False
    
    @staticmethod
    def ecludianNorm(X, C):
        return np.linalg.norm(X-C, axis=1, keepdims=True)
    
    @staticmethod
    def cosineNorm(X, C):
        C = C.reshape(1, -1)
        return np.sum((X * C), axis=1, keepdims=True) / (np.linalg.norm(X, axis=1, keepdims=True) * np.linalg.norm(C, axis=1, keepdims=True))
    
        