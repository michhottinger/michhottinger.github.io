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
