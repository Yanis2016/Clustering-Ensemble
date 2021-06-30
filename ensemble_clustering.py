#coding:utf-8

#Copyright (C) 2020, Mr.Yanis AIT HAMMOU, contact : (yanis.aithammou@outlook.fr)

import numpy as np
import matplotlib.pyplot as plt 
from itertools import combinations
import pandas as pd
from  scipy.spatial.distance import cdist
from ClusteringNormalizedCuts import ClusteringNormalizedCuts
from functools import partial
import multiprocessing as mlp

class KMeans:
    """
        K-Means clustering.
        
        Parameters
        ----------
        n_cluters : int, default=4
            The number of clusters to form as well as the number of centroids to generate.
            
        init : {'k-means++', 'random'}, default='k-means++'
            Method for initialization, defaults to 'k-means++':
            
            'k-means++' : selects initial cluster centers for k-mean 
            clustering in a smart way to speed up convergence.
            
            'random': choose k observations (rows) at random from data for
            the initial centroids.
 
        max_iter : int, default=100
            Maximum number of iterations of the k-means algorithm for a single run.
                 
        dist : str, default='euclidean'
            The distance metric to use.
        
        Attributes
        ----------
        cluster_centers_ : ndarray of shape (n_clusters, n_features)
            Coordiantes of cluster center.
        
        labels_ : ndarray of shape (n_samples, )
            Labels of each point.
        
        inertia_ : float
            Sum of squared distances of samples to their closest cluster center.
            
        n_iter_ : int
            Number of iterations run.   
    """
    
    
    def __init__(self, n_cluters=4, init='k-means++', max_iter=100, dist="euclidean"):
        """
            Initialize self.
        """
        self.__n_cluters = n_cluters
        self.__init = init
        self.__max_iter = max_iter
        self.__dist = dist

    def __initialisation(self, X):
        if self.__init == 'k-means++':
            self.cluster_centers_ = X[np.random.randint(0, X.shape[0], 1), :]
            for i in range(self.__n_cluters-1):
                dist = np.min(cdist(X, self.cluster_centers_, self.__dist), axis=1)
                self.cluster_centers_ = np.concatenate([self.cluster_centers_, 
                                                       X[np.argmax(dist), :].reshape(1,X.shape[1])], 
                                                       axis=0)
        else:
            self.cluster_centers = X[np.random.randint(0, X.shape[0], 3), :]

    def __affecte_cluster(self, X):
        dist = cdist(X, self.cluster_centers_, self.__dist)
        return np.argmin(dist, axis=1), np.min(dist, axis=1)

    def __nouveaux_centroides(self, X):
        return np.array([list((X[self.labels_ == i]).mean(axis=0)) for i in np.unique(self.labels_)])

    def __inertie_globale(self, cluster_dist):
        return np.sum([np.power(cluster_dist[self.labels_ == i], 2).sum() for i in np.unique(self.labels_)])


    def fit(self, X):
        """
            Compute k-means clustering.
            
            Parameters
            ----------
            X : array, shape=(n_samples, n_features)
                Training instances to cluster.
            
            Returns 
            -------
            self
                Fitted estimator.
        """
        self.__initialisation(X)
        self.inertia_ = float('inf')
        self.n_iter_ = 0
        for i in range(self.__max_iter):
            self.n_iter_ += 1
            if self.n_iter_ != 1:
                self.cluster_centers_ = self.__nouveaux_centroides(X)
            self.labels_, cluster_dist = self.__affecte_cluster(X)
            new_inertia = self.__inertie_globale(cluster_dist)
            if new_inertia == self.inertia_:
                self.inertia_ = new_inertia
                break
        return self
                
    def fit_predict(self, X):
        """
            Compute cluster centers and predict cluster index for each sample.
            
            Parameters
            ----------
            X : array, shape=(n_samples, n_features)
            
            Returns
            -------
            labels : array, shape [n_samples,]
                Index of the cluster each sample belongs to.
            
        """
        return self.fit(X).labels_

    
    
def average_confidence(CM, labels):
    """
        Compute the Average Confidence of assignment of the objects to its clusters.
         
        Parameters
        ----------
        CM : ndarray of shape(n_samples, n_samples)
            Co-association matrix.
         
        labels : ndarray of shape(n_samples, )
            Labels of clustering.
         
        Return
        ------
        float
            Average confidence of assignment of the objects to its clusters.
    """
    # compute the degree of confidence of assigning an object xi to its cluster Cp.
    n_cluster = np.unique(labels).size
    conf = np.zeros(labels.size)
    for xi in range(labels.size):
        xi_lab = labels[xi]
        smc = CM[xi, (labels == xi_lab) & (np.arange(labels.size) != xi)]
        if smc.size == 0 :
            smc = 0
        else: 
            smc = smc.mean()
        smac = []
        for j in range(n_cluster):
            if j != labels[xi]:
                smac.append(CM[xi, labels == j].mean())
        if len(smac) == 0:
            conf[xi] = smc
        else:
            conf[xi] = (smc - np.max(smac))
    return conf.mean()


def average_neighborhood_confidence(CM, labels, m=3):
    """
        Compute the Average Neighborhood Confidence of assigning the objects to its clusters.
  
        Parameters
        ----------
        CM : ndarray of shape(n_samples, n_samples)
            Co-association matrix.
         
        labels : ndarray of shape(n_samples, )
            Labels of clustering.
            
        m : int, default=3
            Number of neighbors.
         
        Return
        ------
         float
             Average  neighborhood confidence of assignment of the objects to its clusters.   
    """
    n_cluster = np.unique(labels).size
    conf = np.zeros(labels.size)
    if m == 0:
        return conf 
    for xi in range(labels.size):
        xi_lab = labels[xi]
        smc = CM[xi, (labels == xi_lab) & (np.arange(labels.size) != xi)]
        if smc.size == 0 :
            smc = 0
        elif smc.size >= m:
            smc = np.sort(smc)[::-1][:m].mean()
        else:
            smc = smc.mean()
        smac = []
        for j in range(n_cluster):
            if j != labels[xi]:
                smac_ = CM[xi, labels == j]
                if smac_.size > 0:
                    if smac_.size >= m:
                        smac_ = np.sort(smac_)[::-1][:m]
                    smac.append(smac_.mean())
        if len(smac) == 0:
            conf[xi] = smc
        else:
            conf[xi] = (smc - np.max(smac))
    return conf.mean()


def average_dynamique_neighborhood_confidence(CM, labels, alpha=0.5):
    """
        Compute the Average Dynamic Neighborhood Confidence of assigning the objects to its clusters.
        
        Parameters
        ----------
        CM : ndarray of shape(n_samples, n_samples)
            Co-association matrix.
         
        labels : ndarray of shape(n_samples, )
            Labels of clustering.
            
        alpha : float, default=0.5
            Use to calculate the number of neighbors dynamically.
         
        Return
        ------
         float
             Average dynamic neighborhood confidence of assigning the objects to its clusters.
    """
    n_cluster = np.unique(labels).size
    conf = np.zeros(labels.size)
    m = 1
    for xi in range(labels.size):
        xi_lab = labels[xi]
        smc = CM[xi, (labels == xi_lab) & (np.arange(labels.size) != xi)]
        m = int(np.floor(alpha * smc.sum()))
        if m == 0:
            m = 1
        if smc.size == 0 :
            smc = 0
        elif smc.size >= m:
            smc = np.sort(smc)[::-1][:m].mean()
        else:
            smc = smc.mean()
        smac = []
        for j in range(n_cluster):
            if j != labels[xi]:
                smac_ = CM[xi, labels == j]
                if smac_.size > 0:
                    if smac_.size >= m:
                        smac_ = np.sort(smac_)[::-1][:m]
                    smac.append(smac_.mean())
        if len(smac) == 0:
            conf[xi] = smc
        else:
            conf[xi] = (smc - np.max(smac))
    return conf.mean()


def vat(CM):
    """
        Compute the matrix for Visual Assessment of cluster Tendency.
        
        Parameters
        ----------
        CM : ndarray of shape(n_samlpes, n_samples)
            Co-association matrix, dissimilarity data.
        
        Return
        ------
        CM_VAT : ndarray of shape(n_samples, n_samples)
            Reorder dissimilarity data.
        
        Q : ndarray of shape (n_samples, )
            Reorder indexs of CM.
    """
    J = np.zeros((CM.shape), dtype=bool)
    I = np.ones((CM.shape), dtype=bool)
    Q = []
    cm_mask = None
    for t in range(CM.shape[0]):
        if t == 0:
            cm_mask = np.ma.masked_array(CM, mask = (I&J))
        else:
            cm_mask = np.ma.masked_array(CM, mask = (I|J))    
        i, j = np.unravel_index(cm_mask.argmin(), CM.shape)
        Q.append(j)
        J[:, j] = True
        I[j, :] = False
    cm_vat = CM.copy()
    for i in range(CM.shape[0]):
        for j in range(CM.shape[0]):
            cm_vat[i, j] = CM[Q[i], Q[j]]
            cm_vat[j, i] = CM[Q[i], Q[j]]
    return cm_vat, np.array(Q)


def gen_base_partition(X, k, It):
    return KMeans(k, max_iter=It).fit_predict(X)


def gen_base_partition_by_kmeans(X, M=10, It=4, Ktype='Random'):
    """
        Generate base partition by k-means
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            data set.
        
        M : int default=10
            The number of base partitions
        
        It : int, default=10
            The number of iterations for k-means
        
        Ktype : {'Fixed', 'Random'}, default=Random
            The type to generate base partitions, 'Fixed' : k = sqrt(N)
        
        Return
        ------
        PI : ndarray of shape(n_sample, n_partitions)
            Base partitions, one column is a partition
    """
    N = X.shape[0]
        
    PI = []
    CM =  CM = np.zeros((X.shape[0], X.shape[0]))
    K = 0
    if Ktype == 'Fixed':
        k = [int(np.ceil(np.sqrt(N)))]*M
    else:
        k = np.random.randint(2, np.ceil(np.sqrt(N)), M)
    kmeans = partial(gen_base_partition, X, It=It)
    
    pool = mlp.Pool(mlp.cpu_count())
    
    PI = pool.map(kmeans, k)
    pool.close()
    pool.join()

    return np.transpose(PI)



def gen_cm(PI):
    """
        Generate Co-association Matrix
        
        Parameters
        ----------
        PI : ndarray of shape(n_samples, n_partitions)
            Base partitions, one column is a partition.
            
        Return
        ------
        CM : ndarray of shape(n_samples, n_samples)
            Co-association matrice.
    """
    n_samples = PI.shape[0]
    n_partitions = PI.shape[1]
    cm = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        cm[i, :] = (PI == PI[i, :]).sum(axis=1)
    return cm/n_partitions

class EnsembleClustering:
    
    def __init__(self, n_clusters=4, n_partitions=1000, max_iter=4, k_type='Fixed', cons_validation='ac', m=3, alpha=0.5):
        """
            Clustering Ensemble.
            
            Operating principle:
            ---------
                1 - generation of several partitions using the K Means algorithm with k 
                    varies between 2 and sqrt (n_sample).

                2 - update of the co-association matrix.

                3 - generation of partitions by applying the clustering algorithm based on normalized cuts 
                    on the co_assiation matrix when the negative proof is removed.

                4 - selection of the final score with a higher degree of confidence.
            
            Parameters
            ----------
            n_cluters : int, default=4
                The number of clusters.

            n_partitions : int, default=1000
                Number of base partitions.

            max_iter : int, default=100
                Maximum number of iterations of the k-means algorithm for a single run.

            k_type : {'Fixed', 'Random'}, default='Fixed'
                The type to generate base partitions, 
            'Fixed' : k = sqrt(n_sample),
            'Random': k = random between 2 and sqrt (sample n).

            cons_validation: {'ac', 'anc', 'andc'}, default='ac'
                Type of method to use to select the final partition.
            'ac' : average Confidence of assignment of the objects to its clusters. 
            'anc': average Neighborhood Confidence of assigning the objects to its clusters.
            'andc': average dynamic neighborhood confidence of assigning the objects to its clusters.

            m : int, default=3
                Number of neighbors, used when cons_validation = 'anc'.

            alpha : float > 0, default=0.5
                Use when cons_validation = 'etc' to dynamically calculate the number of neighbors.


            Attributes
            ----------
            co_association_matrix : ndarray of shape (n_clusters, n_samples)
                Co-association matrix.

            labels_ : ndarray of shape (n_samples, )
                Labels of each point, corresponds to the best partition selected by
                one of the methods {'ac', 'anc', 'adnc'}

            partitions : ndarray of shape(n_samples, 50)
                Partitions generated by the ncut algorithm by removing negative evidence
                from the co-association matrix.
            
            quality_of_partitions : ndarray of shape (50).
                Quality measured on each partition 
                by one of the evaluation methods {'ac', 'anc', 'adnc'}
        """
        self.__n_clusters = n_clusters
        self.__n_partitions = 1000
        self.__max_iter = max_iter
        self.__k_type = k_type
        self.__cons_validation = cons_validation
        self.__m = m
        self.__alpha = alpha
        self.co_association_matrix = []
        self.labels = None
        
    def fit(self, X):
        """
            Compute clusters.
            
            Parameters
            ----------
            X : array, shape=(n_samples, n_features)
            
            Returns 
            -------
            self
                Fitted estimator.
        """
        if type(X) == pd.DataFrame:
            X = np.asarray(X)
        
        if len(self.co_association_matrix) == 0:
            PI = gen_base_partition_by_kmeans(X, It=self.__max_iter, M=self.__n_partitions, Ktype=self.__k_type)
            self.co_association_matrix = gen_cm(PI)
        self.partitions = []
        for i in np.arange(0.01, 0.51, 0.01):
            CM_ = self.co_association_matrix.copy()
            partition = ClusteringNormalizedCuts(self.__n_clusters).fit_predict(CM_)
            self.partitions.append(partition)
        self.partitions = np.transpose(self.partitions)
        
        self.quality_of_partitions = []
        if self.__cons_validation == 'ac':
            self.quality_of_partitions.append(np.apply_along_axis(lambda x: average_confidence(self.co_association_matrix,  x),
                                     0, self.partitions))
        elif self.__cons_validation == 'anc':
            self.quality_of_partitions.append(np.apply_along_axis(lambda x: average_neighborhood_confidence(self.co_association_matrix, x,
                                                                      self.__m),
                                          0, self.partitions))
        else:
            self.quality_of_partitions.append(np.apply_along_axis(lambda x: average_dynamique_neighborhood_confidence(self.co_association_matrix,
                                                                                              x, self.__alpha),
                                          0, self.partitions))
        self.quality_of_partitions = np.array(self.quality_of_partitions).ravel()
        self.labels = self.partitions[:, self.quality_of_partitions.argmax()].astype(np.int)
        
        return self
        
    def fit_predict(self, X):
        """
            Compute clusters  and predict clustered index for each sample.
            
            Parameters
            ----------
            X : array, shape=(n_samples, n_features)
            
            Returns
            -------
            labels : array, shape [n_samples,]
                Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels
    
    def predict(self, X):
        """
            Predict the closest cluster each sample in X belongs to.
            
            Parameters
            ----------
             X : array, shape=(n_samples, n_features).

            Returns
            -------
            labels : array, shape [n_samples,]
                Index of the cluster each sample belongs to.
        """
        if self.labels is None:
            raise NotImplementedError("This EnsembleCLustering instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        return self.labels
    
    
    def draw_vat(self, figsize=(10, 10)):
        """
            Calculate and display the matrix for visual assessment of the cluster trend.
            
            Parameters
            ----------
            figsize : tuple, default = (10, 10)
                The dimensions of the figure.
        """
        if len(self.co_association_matrix) == 0:
            PI = gen_base_partition_by_kmeans(X, It=self.__max_iter, M=self.__n_partitions, Ktype=self.__k_type)
            self.co_association_matrix = gen_cm(PI)        
        vat_cm, index = vat(self.co_association_matrix)
        plt.figure(figsize=figsize)
        plt.title("VAT")
        im = plt.imshow(vat_cm, cmap='seismic')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()
