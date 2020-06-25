---
layout: post
title: K-means Clustering for Analysis of Heart Disease
subtitle: "How K-Means can help uncover information in unsupervised learning"
#gh-repo: daattali/beautiful-jekyll
#gh-badge: [star, fork, follow]
tags: [kmeans, unsupervised machine learning, heart disease]
comments: true
---

This article was written to help understand the power of K-means in predicting health outcomes in unsupervised machine learning. 

**[Read the full article here](https://medium.com/@michellibelly/k-means-clustering-for-analysis-of-heart-disease-c2c6f75927e0).**

**[Source Code Here](https://github.com/michhottinger/CS-Data-Science-Build-Week-1)**



K-Means Clustering for Analysis of Heart Disease
How K-Means can help uncover information in unsupervised learning

![image](https://github.com/michhottinger/michhottinger.github.io/blob/master/img/heartdisease.jpg)

photo cred: https://health.clevelandclinic.org/is-heart-disease-curable/

Have you ever wondered what is meant to be done with the abundance of data that is floating around. Many datasets lack a column that answers a specific question such as , "does this person have cancer" or "is this person a yes or a no". The outcomes label (also known as target or Y-label) can answer a question that summarizes of all the data. The classic Titanic dataset for example, has several informative columns about ticket price, age, gender, family onboard, but the column that sums this all together in a "so what" is the "survived" column that states if the person lived or died on the Titanic. We can use the "survived" column as a target and determine which other columns of data may have resulted in a survival versus a death. This is supervised learning. We have an abundance of health data, but we don't know if the person has a disease, or lives or dies yet. Without this target data, this is a project for unsupervised machine learning. A great place to start an unsupervised project is with K-means clustering. Using K-means, we are looking at ways in which the data is grouped. We can start with an arbitrary number of K clusters (let's start with three) and calculate the lowest sum of squares error each data point has from the nearest cluster center (known as a centroid). We want the three centroids to be as far apart from one another while also being as close to their respective data points as possible.

![image](https://github.com/michhottinger/michhottinger.github.io/blob/master/img/Screen%20Shot%202020-06-25%20at%2012.48.01.png)

Centroids spaced apart while clustered within each colored group

Once we have our three clusters, we are going to run this process again and again until the sum of squares error is the same as the lowest result every time and can no longer update to a better result. When we first place our centroids in the data, it is arbitrary where they go, so the repetition is to find the optimal placement to maximize centroid distances while minimizing the sum of squares error for each datapoint. I coded my own K-means function and also used the K-means function found in the Scikit-learn library. I recommend using the library since it has numerous methods that are useful beyond simple K-means. Since placement of centroids is random, you will notice a difference in the place of my centroids, yet the results are the same.

![image](https://github.com/michhottinger/michhottinger.github.io/blob/master/img/KMeans.png)
![image](https://github.com/michhottinger/michhottinger.github.io/blob/master/img/Screen%20Shot%202020-06-25%20at%2012.48.01.png)

On the left is the K-means from my own implementation, while the right utilizes Scikit Learn K-means. The centroids are placed differently with different clusters.

With the three beautiful clusters, we can start to ask ourselves, what is going on with our data features in each of these clusters? We can take the labels from our clusters (color-coded) and append those labels to our dataset then group the data according to the new labels to determine if there are any new relationships that were not visible previously.

![image](https://github.com/michhottinger/michhottinger.github.io/blob/master/img/Screen%20Shot%202020-06-25%20at%2012.53.02.png)
Labels from k-means clusters applied to the entire dataset and graphing the age and chol with groups. 

The heart dataset I used is full of numeric categorical data, which does not cluster well as a whole. To solve this issue and tailor the machine learning to the data, I should transition to a K-modes function for machine learning rather than the K-means. K-modes replaces the means of clusters with modes and works in a similar fashion to K-means.

![image](https://github.com/michhottinger/michhottinger.github.io/blob/master/img/Screen%20Shot%202020-06-25%20at%2012.57.58.png)

No nice clusters on the entire dataset. Too many dimensions and no clear groups for K-means.

To overcome some obstacles with the data type, I chose to use only the continuous numerical data, create two cluster groups and apply those cluster labels to the data. 

![image](https://github.com/michhottinger/michhottinger.github.io/blob/master/img/Screen%20Shot%202020-06-25%20at%2014.31.19.png)

Scaled multidimensional data clusters on heart disease.

Since this dataset happens to contain a target feature, I have the opportunity to check the accuracy of my k-means clusters. In this heart data, the target indicates if the patient had heart disease [1] or does not have heart disease [0]. The accuracy of my self-made k-means was 74.59% while the accuracy of Sci-kit Learn's k-means was 74.26%. The difference is likely due to the initialization position of the centroids in the data.

![image](https://github.com/michhottinger/michhottinger.github.io/blob/master/img/Screen%20Shot%202020-06-25%20at%2014.08.36.png)
![image](https://github.com/michhottinger/michhottinger.github.io/blob/master/img/Screen%20Shot%202020-06-25%20at%2014.10.02.png)

Dataset on the left shows the k-means clusters while the dataset on the right shows the actual target values for no disease (0) Green or disease (1) red. Magenta and red are comparable clusters that signify >50% diameter narrowing and is an angiographic disease status of heart disease.

When attempting to predict if a person will have heart disease, we can see based on our above graphs that the age and cholesterol are not generally a good predictor. See the green and blue data points without disease? Greater age does not increase risk. Cholesterol levels are not indicative either, otherwise we would see a pattern of upward red. Most of the red is right in the middle scattered upward and downward. Two better indicators for disease can be observed below.

![image](https://github.com/michhottinger/michhottinger.github.io/blob/master/img/Screen%20Shot%202020-06-25%20at%2014.47.04.png)

Thalach: Maximum heart rate achieved and Oldpeak: Stress Test: depression induced by exercise relative to rest

With the above data provided, we can draw a clear slanted line and see that a patient will either have heart disease or not, regardless of age, weight, gender, cholesterol and the several other features we have available. In this case, magenta is disease [0] while blue is disease free. Oldpeak (a stress test measurement) would not indicate much on its own, but when graphed next to Thalach (max heart rate) we can create a nice divide using the labels from our clusters (blue and magenta). If a patient falls on the magenta side of the line, they are very likely to have heart disease. If they fall on the blue side, they are likely to be disease free, for now.
K-means clusters are a nice way to visualize data when we are not sure what we are looking for. Finding clusters then labeling the data with the cluster labels to create your own "target" feature is a great way to handle unlabeled data for unsupervised machine learning. As you can see, once application can be finding hidden features that may indicate a disease state in patients. Finding the features that most accurately indicate a given disease can save both money and lives.

Below I will post my code for implementing the above study.
```
import numpy as np
from numpy.linalg import norm


class Kmeans:
    '''Implementing Kmeans algorithm.'''

    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initialize_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance
    def find_closest_cluster(self, distance):
#Returns the indices of the minimum values along an axis
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
#sum of squares Error calc
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance=self.compute_distance(X,old_centroids)
            self.labels=self.find_closest_cluster(distance)
            self.centroids=self.compute_centroids(X,self.labels)
            if np.all(old_centroids == self.centroids):
#Test whether all array elements along a given axis evaluate to True
#this is looking for convergence
                break
        self.error = self.compute_sse(X, self.labels,self.centroids)
    
    def predict(self, X):
        old_centroids = self.centroids#to define old within function
        distance = self.compute_distance(X, old_centroids)
        return self.find_closest_cluster(distance)
```

Next we will read in dataset, clean data and visualize the data.
```
# Import the data
df = pd.read_csv('https://raw.githubusercontent.com/michhottinger/CS-Data-Science-Build-Week-1/master/datasets_33180_43520_heart.csv')
df.head(5)
#dealing with categorical data. One hot encoding would work here too

a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")
d = pd.get_dummies(df['sex'], prefix = "sex")
e = pd.get_dummies(df['restecg'], prefix = "restecg")

frames = [df, a, b, c, d, e]
df = pd.concat(frames, axis = 1)



df_copy = df.drop(columns = ['cp', 'thal', 'slope', 'sex', 'restecg'])
df_copy.head()
#very important to drop the target if it is present
df_drop = df.drop(columns = ['target'])
df_drop.head(5)
# Plot the data
plt.figure(figsize=(6, 6))
features = ['age',	'sex',	'trestbps',	'chol',	'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'ca', 'target', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'thal_0', 'thal_1', 'thal_2', 'thal_3', 'slope_0',	'slope_1', 'slope_2',	'cp_0',	'cp_1',	'cp_2',	'cp_3',	'thal_0',	'thal_1',	'thal_2',	'thal_3', 'slope_0', 'slope_1', 'slope_2']
X = df_copy['age']
y = df_copy['chol']
plt.scatter(X, y)
plt.xlabel('')
plt.ylabel('')
plt.title('Visualization of raw data');
#use a subset of the data to start k-means exploration
data = df_copy[['age', 'chol']]
```
Now use your K-means to get labels and view data.
```
# Standardize the data
X_std = StandardScaler().fit_transform(data)


# Run local implementation of kmeans Here we tested 3 clusters
km = Kmeans(n_clusters=3, max_iter=100, random_state = 42)
km.fit(X_std)
centroids = km.centroids
# labels_ in Scikit Learn are equivalent to calling fit(x) then predict
labels_ = km.predict(X_std)
labels_
# Plot the clustered data
fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(X_std[labels_ == 0, 0], X_std[labels_ == 0, 1],
            c='green', label='cluster 1')
plt.scatter(X_std[labels_ == 1, 0], X_std[labels_ == 1, 1],
            c='blue', label='cluster 2')
plt.scatter(X_std[labels_ == 2, 0], X_std[labels_ == 2, 1],
            c='yellow', label='cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,
            c='r', label='centroid')
plt.legend()
plt.xlim([-3, 4])
plt.ylim([-3, 4])
plt.xlabel('age')
plt.ylabel('chol')
plt.title('Visualization of clustered data', fontweight='bold')
ax.set_aspect('equal');
```
Check out the labels and put them on the dataset.
```
#labels added to dataset
data['cluster'] = labels_
data.head(5)
#uses lables from clusters to see on data
fig, ax = plt.subplots()
colors = {0:'red', 1:'blue', 2:'yellow'}
grouped = data.groupby('cluster')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='age', y='chol', label=key, color=colors[key])
plt.show()
```
To determine the best number of clusters, you can also use the elbow method.
```
#elbow method:
# Run the Kmeans algorithm and get the index of data points clusters
sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X_std)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Inertia');
```
Now you have seen all the code to run your own K-means testing. Enjoy!
