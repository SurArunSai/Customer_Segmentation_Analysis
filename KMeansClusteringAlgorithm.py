import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    confusion_matrix,
    accuracy_score,
    classification_report,
    log_loss,
)
from math import sqrt
from sklearn.metrics import silhouette_samples, silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from flask import Flask, render_template, request
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.cluster import KMeans, k_means

class KMeansClusteringAlgorithm:
    """
    This class provides methods to analyze and cluster data.

    Methods:
    - __init__(data_path): Initialize the KMeansClusteringAlgorithm object with the path to the data file.
    - import_raw_data(): Import the raw data from the specified file.
    - preprocess_data(): Preprocess the raw data by performing one-hot encoding and normalization.
    - random_centroids(k): Generate random initial centroids for clustering.
    - get_labels(centroids): Assign labels to data points based on the closest centroid.
    - new_centroids(labels, k): Update centroids based on the mean of data points in each cluster.
    - cluster_data(max_iterations, centroid_count): Perform clustering on the data.
    """

    def __init__(self, data_path):
        """
        Initialize the DataAnalyzer object with the path to the data file.

        Parameters:
        - data_path (str): The path to the data file.

        Returns:
        None
        """
        self.data_path = data_path
        self.raw_data = pd.read_csv(self.data_path, sep=";")
        self.copy_raw_data = pd.read_csv(self.data_path, sep=";")
        self.features = [
            'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'previous', 'poutcome', 'day_of_week', 'y'
        ]
        self.new_raw_data = None
        self.df = None

    def import_raw_data(self):
        """
        Import the raw data from the specified file.

        Parameters:
        None

        Returns:
        None
        """
        self.raw_data = pd.read_csv(self.data_path, sep=";")

    def preprocess_data(self):
        """
        Preprocess the raw data by performing one-hot encoding and normalization.

        Parameters:
        None

        Returns:
        None
        """
        self.new_raw_data = pd.get_dummies(self.raw_data, columns=self.features)
        self.df = self.new_raw_data.dropna()
        self.new_raw_data = (
            (self.new_raw_data - self.new_raw_data.min()) /
            (self.new_raw_data.max() - self.new_raw_data.min())
        ) * 9 + 1

    def random_centroids(self, k):
        """
        Generate random initial centroids for clustering.

        Parameters:
        - k (int): The number of centroids to generate.

        Returns:
        pandas.DataFrame: Randomly generated centroids.
        """
        centroids = []
        for i in range(k):
            centroid = self.new_raw_data.apply(lambda x: float(x.sample()))
            centroids.append(centroid)
        return pd.concat(centroids, axis=1)

    def get_labels(self, centroids):
        """
        Assign labels to data points based on the closest centroid.

        Parameters:
        - centroids (pandas.DataFrame): The centroids for clustering.

        Returns:
        pandas.Series: Cluster labels for each data point.
        """
        distances = centroids.apply(
            lambda x: np.sqrt(((self.new_raw_data - x) ** 2).sum(axis=1))
        )
        return distances.idxmin(axis=1)

    def new_centroids(self, labels, k):
        """
        Update centroids based on the mean of data points in each cluster.

        Parameters:
        - labels (pandas.Series): Cluster labels for each data point.
        - k (int): The number of clusters.

        Returns:
        pandas.DataFrame: Updated centroids.
        """
        centroids = self.new_raw_data.groupby(labels).apply(
            lambda x: np.exp(np.log(x).mean())
        ).T
        return centroids

    def cluster_data(self, max_iterations, centroid_count):
        """
        Perform clustering on the data.

        Parameters:
        - max_iterations (int): The maximum number of iterations for clustering.
        - centroid_count (int): The number of centroids/clusters.

        Returns:
        tuple: A tuple containing the cluster labels and the data with cluster labels assigned.
        """
        centroids = self.random_centroids(centroid_count)
        old_centroids = pd.DataFrame()
        iteration = 1

        while iteration < max_iterations and not centroids.equals(old_centroids):
            old_centroids = centroids

            labels = self.get_labels(centroids)
            centroids = self.new_centroids(labels, centroid_count)
            iteration += 1

        self.copy_raw_data['cluster'] = labels  # Add cluster labels to the df DataFrame

        return labels, self.copy_raw_data
