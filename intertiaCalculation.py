def calculate_inertia(df, labels, centroids):
    # get the number of clusters
    k = centroids.shape[0]
    
    # initialize an empty array to store inertia scores
    inertia = np.zeros(k)
    
    # iterate over each cluster
    for i in range(k):
        # get the data points for this cluster
        cluster_points = df[labels == i]
        
        # get the distance of each point to the centroid and square it
        distances = np.linalg.norm(cluster_points - centroids.iloc[i], axis=1) ** 2
        
        # sum the distances to get the inertia for this cluster
        inertia[i] = np.sum(distances)
        
    return inertia


# assuming data is in a pandas dataframe called 'df'
def calculate_inertia_for_range_of_clusters(df, max_clusters):
    # initialize an empty list to store inertia scores
    inertia_scores = []
    
    # iterate over each cluster size from 2 to max_clusters
    for k in range(2, max_clusters + 1):
        # randomly initialize k centroids
        centroids = df.sample(n=k)
        
        # initialize an array to store the labels for each data point
        labels = np.zeros(df.shape[0])
        
        # iterate until the labels no longer change
        while True:
            # calculate the distance of each data point to each centroid
            distances = np.linalg.norm(df.values[:, np.newaxis, :] - centroids.values, axis=2)
            
            # assign each data point to the closest centroid
            new_labels = np.argmin(distances, axis=1)
            
            # if the labels haven't changed, break out of the loop
            if np.array_equal(labels, new_labels):
                break
            
            # update the labels
            labels = new_labels
            
            # update the centroids
            for i in range(k):
                centroids.iloc[i] = df[labels == i].mean()
                
        # calculate the inertia for each cluster
        inertia = calculate_inertia(df, labels, centroids)
        
        # sum the inertia scores to get the total inertia for this cluster size
        total_inertia = np.sum(inertia)
        
        # add the total inertia to the list of inertia scores
        inertia_scores.append(total_inertia)
        
        # print the inertia for this cluster size
        print("The inertia for", k, "clusters is:", total_inertia)
        
    return inertia_scores


inertia_scores = calculate_inertia_for_range_of_clusters(new_raw_data, 19)

plt.plot(range(2,20), inertia_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Scree Plot for Inertia - Elbow Method')
plt.show()