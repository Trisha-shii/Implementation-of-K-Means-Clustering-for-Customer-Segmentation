# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1.Import the necessary packages using import statement.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Import KMeans and use for loop to cluster the data.

4.Predict the cluster and plot data graphs.

5.Print the outputs and end the program
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: TRISHA PRIYADARSHNI PARIDA
RegisterNumber:  212224230293
*/


import pandas as pd
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

data = pd.read_csv("Mall_Customers.csv")

data.head()

data.info()

data.isnull().sum()


X = data.iloc[:, [3, 4]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()


X=data[["Annual Income (k$)","Spending Score (1-100)"]]
X

k =5
kmeans=KMeans(n_clusters=k)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels=kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)



X = df.iloc[:, [3, 4]].values  
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids', edgecolor='black')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()


```

## Output:

![Screenshot 2025-05-05 105901](https://github.com/user-attachments/assets/0cb62f5d-e71d-4f90-b12e-681938dee02d)

![Screenshot 2025-05-05 105907](https://github.com/user-attachments/assets/1da55141-4ab3-4b4b-83ca-6c398f9ba63d)

![Screenshot 2025-05-05 105911](https://github.com/user-attachments/assets/27d3f091-dee7-4e25-9086-152ef70ade0b)


![Screenshot 2025-05-05 105849](https://github.com/user-attachments/assets/f2e31cec-a5fc-4bc6-879a-dcd84b877d1a)


![Screenshot 2025-05-05 105755](https://github.com/user-attachments/assets/61367291-8287-4388-a010-9395326ba234)


![Screenshot 2025-05-05 105809](https://github.com/user-attachments/assets/68cb3050-f214-4e9e-8e55-963476034a36)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
