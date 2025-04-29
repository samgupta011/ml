
# K-Means Clustering Algorithm - Iris Dataset

This project demonstrates the implementation of the **K-Means clustering algorithm** using the **Iris dataset**. The K-Means algorithm groups data into **K** clusters based on feature similarity. In this case, we use the Iris dataset, which contains 150 data points and 4 features, to apply and visualize K-Means clustering.

---

## 📦 Requirements

You can install the necessary dependencies using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## 🧠 Overview of K-Means Clustering

The **K-Means clustering algorithm** is an unsupervised learning technique used for grouping similar data points into clusters. The number of clusters is predefined as `K`. The algorithm works by:
1. Assigning each data point to the nearest cluster.
2. Recalculating the centroids of each cluster.
3. Iterating these steps until convergence.

In this program, we apply K-Means clustering on the **Iris dataset** with `K = 3` (since the Iris dataset has three species of flowers).

---

## 📁 Code Explanation

### 1. **Importing Libraries**

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
```

- `numpy` and `pandas` are used for data manipulation and handling.
- `KMeans` from `sklearn.cluster` implements the K-Means clustering algorithm.
- `datasets` from `sklearn` provides built-in datasets, like the Iris dataset.
- `silhouette_score` from `sklearn.metrics` is used to evaluate the clustering quality.
- `matplotlib.pyplot` is used for plotting the clustering results.

---

### 2. **Loading the Dataset**

```python
iris = datasets.load_iris()
```

- The **Iris dataset** is loaded using `datasets.load_iris()`, which contains 150 samples of Iris flowers, each described by four features (sepal length, sepal width, petal length, and petal width).

---

### 3. **Creating a DataFrame**

```python
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
```

- We create a **pandas DataFrame** from the Iris dataset, where `iris.data` contains the feature values, and `iris.feature_names` are the column names.

---

### 4. **Preparing Data for Clustering**

```python
X = df.values
```

- The feature data (`X`) is extracted from the DataFrame for clustering. Here, we use all the features for clustering.

---

### 5. **Applying K-Means Clustering**

```python
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
```

- We initialize the `KMeans` object with `n_clusters=3` (since there are three species in the Iris dataset).
- The `.fit()` method is used to perform the clustering by training the K-Means algorithm on the feature data (`X`).

---

### 6. **Getting Clustering Results**

```python
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
```

- The centroids of the clusters are obtained using `kmeans.cluster_centers_`.
- The labels (cluster assignments) for each data point are obtained using `kmeans.labels_`.

---

### 7. **Evaluating Clustering Performance**

```python
silhouette_avg = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette_avg:.2f}")
```

- The **Silhouette Score** is calculated using `silhouette_score(X, labels)`, which helps assess how well the data points have been clustered. The score ranges from -1 to 1, with a higher score indicating better-defined clusters.

---

### 8. **Visualizing the Clusters**

```python
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering of Iris Dataset")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.show()
```

- We use `matplotlib` to create a scatter plot of the clustered data points. The points are colored based on their cluster label.
- The centroids of the clusters are marked with red "X" markers.
- The plot uses the first two features of the dataset for visualization.

---

## 📊 Output

The program will display the following output:

```text
Silhouette Score: 0.46
```

It will also display a plot showing:
- **Data points**: Represented as colored dots, indicating the cluster to which each data point belongs.
- **Cluster centroids**: Red "X" markers that represent the center of each cluster.

---

## 💡 Key Takeaways

- **K-Means Clustering**: A simple and widely used algorithm for clustering data points based on similarity.
- **Silhouette Score**: A metric that helps evaluate the quality of clustering. A higher silhouette score indicates well-separated clusters.
- **Visualization**: The plot helps to visually interpret the clustering results and understand how well the data points are grouped.

---

## 🚀 How to Run

1. Install the required dependencies (`numpy`, `pandas`, `scikit-learn`, and `matplotlib`).
2. Copy the code into a Python file or a Jupyter notebook.
3. Run the script, and it will display the clustering output (Silhouette Score) and the scatter plot with the clusters.

---

## 👨‍💻 Use Case

This project is useful for understanding how K-Means clustering works and how it can be applied to group data into clusters. The Iris dataset provides a simple and clear example, but the method can be applied to much larger and more complex datasets.