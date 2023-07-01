import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class PCACalculator:
    def visualize_pca(self, X, y_num, target_names):
        """
        Perform PCA and visualize the data.

        Parameters:
        - X (pandas.DataFrame): The data points.
        - y_num (numpy.ndarray): The cluster labels for each data point.
        - target_names (list): The names of the target clusters.

        Returns:
        None
        """
        pca = PCA(n_components=2, random_state=453)
        X_r = pca.fit_transform(X)

        # Percentage of variance explained for each component
        print('Explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))

        # Plotting the data
        plt.figure(figsize=(12, 8))
        colors = ['navy', 'turquoise', 'darkorange', 'red', 'black']
        lw = 2

        for color, i, target_name in zip(colors, [0, 1, 2, 3, 4], target_names):
            plt.scatter(X_r[y_num == i, 0], X_r[y_num == i, 1], color=color, alpha=.8, lw=lw, label=target_name)

        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.6)
        plt.title('PCA of 2 Items')
        plt.show()

    def determine_n_components(self, X):
        """
        Determine the number of components needed to explain 95% variance.

        Parameters:
        - X (pandas.DataFrame): The data points.

        Returns:
        - n_components (int): The number of components needed.
        """
        pca = PCA(n_components=X.shape[1], random_state=453)
        X_r = pca.fit_transform(X)

        # Calculating the 95% variance
        total_variance = sum(pca.explained_variance_)
        print("Total Variance in our dataset is:", total_variance)
        var_95 = total_variance * 0.95
        print("The 95% variance we want to have is:", var_95)
        print("")

        # Finding the number of components needed to explain 95% variance
        cumulative_var = pca.explained_variance_[0]
        n_components = 1

        while cumulative_var < var_95:
            cumulative_var += pca.explained_variance_[n_components]
            n_components += 1

        print("Number of components needed to explain 95% variance:", n_components)

        return n_components

    def plot_variance_explained(self, X):
        """
        Plot the variance explained by different numbers of components.

        Parameters:
        - X (pandas.DataFrame): The data points.

        Returns:
        None
        """
        pca = PCA(n_components=X.shape[1], random_state=453)
        X_r = pca.fit_transform(X)

        # Calculating the variance explained by different numbers of components
        explained_variance = []
        n_components_range = [30, 35, 40, 41, 50, 53, 55, 59]
        for n_components in n_components_range:
            explained_variance.append(sum(pca.explained_variance_[:n_components]))

        # Plotting the data
        plt.figure(1, figsize=(14, 8))
        plt.plot(range(X.shape[1]), pca.explained_variance_ratio_, linewidth=2, c="r")
        plt.xlabel('n_components')
        plt.ylabel('explained_ratio_')

        # Plotting line with 95% explained variance
        plt.axvline(59, linestyle=':', label='n_components - 95% explained', c="blue")
        plt.legend(prop=dict(size=12))

        # Adding arrow
        plt.annotate('59 eigenvectors used to explain 95% variance', xy=(59, pca.explained_variance_ratio_[59]),
                     xytext=(57, pca.explained_variance_ratio_[10]),
                     arrowprops=dict(facecolor='blue', shrink=0.05))

        plt.show()


# Example usage
if __name__ == '__main__':
    pca_calculator = PCACalculator()

    X = new_raw_data
    y_num = labels
    target_names = ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"]

    pca_calculator.visualize_pca(X, y_num, target_names)

    n_components = pca_calculator.determine_n_components(X)

    pca_calculator.plot_variance_explained(X)
