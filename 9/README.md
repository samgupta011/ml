
# Agglomerative Clustering - Iris Dataset

This project demonstrates the use of **Agglomerative Clustering**, a hierarchical clustering technique, on the **Iris dataset** using Python. The goal is to group similar data points into clusters without using predefined labels. The Agglomerative Clustering algorithm is applied to cluster the Iris dataset into 3 groups (as there are three types of Iris flowers in the dataset).

---

## 📦 Requirements

To run the code, install the following Python libraries using the `pip` command:

```bash
pip install numpy pandas scikit-learn matplotlib scipy
```

### Libraries:
- **`numpy`**: Used for numerical operations.
- **`pandas`**: For data manipulation and handling datasets.
- **`scikit-learn`**: Machine learning library that includes the `AgglomerativeClustering` algorithm.
- **`matplotlib`**: For plotting graphs and visualizing clustering results.
- **`scipy`**: For hierarchical clustering and generating dendrograms.

---

## 🧠 Overview of Agglomerative Clustering

Agglomerative Clustering is a **bottom-up** hierarchical clustering algorithm. Initially, each data point is considered as its own cluster. The algorithm then merges the closest clusters iteratively until a stopping condition (e.g., the desired number of clusters) is met. In this project, **Agglomerative Clustering** is applied to the Iris dataset, which contains 150 samples of Iris flowers with 4 features each: sepal length, sepal width, petal length, and petal width.

---

## 🖥️ Code Explanation

### 1. **Import Libraries**

```python
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
```

- **`numpy`**: Provides mathematical operations on arrays.
- **`pandas`**: Used to load and manipulate the dataset in DataFrame format.
- **`AgglomerativeClustering`**: Implements the agglomerative clustering algorithm.
- **`datasets`**: Loads built-in datasets (like Iris).
- **`StandardScaler`**: Standardizes features to ensure they are on the same scale.
- **`dendrogram` & `linkage`**: Used for plotting the hierarchical clustering process.

### 2. **Load the Dataset**

```python
iris = datasets.load_iris()
```

- Loads the **Iris dataset** containing features such as sepal length, sepal width, petal length, and petal width.

### 3. **Prepare the Data**

```python
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
X = df.values
```

- Converts the Iris dataset into a pandas DataFrame and extracts the feature values into `X` for clustering.

### 4. **Standardizing the Data**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

- **StandardScaler**: Standardizes the data by removing the mean and scaling to unit variance, ensuring that all features contribute equally to the distance calculation in the clustering algorithm.

### 5. **Perform Agglomerative Clustering**

```python
agg_clust = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_clust.fit_predict(X_scaled)
```

- **`AgglomerativeClustering`**: Performs clustering with **3 clusters**. The `ward` linkage method minimizes the variance of merged clusters and is often preferred when clustering numerical data.

### 6. **Display the Cluster Labels**

```python
print("Cluster labels for each data point:")
print(labels)
```

- Prints the cluster labels assigned to each data point.

### 7. **Plot the Dendrogram**

```python
linked = linkage(X_scaled, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Data points')
plt.ylabel('Euclidean distance')
plt.show()
```

- **Dendrogram**: A tree-like diagram that shows how clusters are merged at each step in the hierarchical process. The y-axis represents the Euclidean distance at which clusters are merged.

### 8. **Visualize the Clustering Results**

```python
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', alpha=0.7)
plt.title("Agglomerative Clustering of Iris Dataset")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
```

- Creates a scatter plot of the first two features of the Iris dataset (sepal length and sepal width) and colors the data points based on their cluster labels.

---

## 📊 Expected Output

- **Cluster Labels**: The cluster labels are printed for each data point in the dataset.
- **Dendrogram**: A hierarchical tree is plotted showing how the clusters are formed.
- **Scatter Plot**: A scatter plot visualizing how the dataset is divided into clusters using Agglomerative Clustering.

---

## 💡 Key Takeaways

- **Agglomerative Clustering** is a **bottom-up** clustering algorithm that merges clusters based on distance.
- The **dendrogram** is a powerful tool to visualize the hierarchical nature of the clustering process.
- Standardizing the data is essential for ensuring that the clustering algorithm performs optimally.
- The results can be visualized using scatter plots to understand how the data points are grouped.

---

## 🚀 How to Run

1. Install the required libraries using the `pip` command mentioned above.
2. Copy the provided code into a Python file or Jupyter notebook.
3. Run the script to view the clustering results, dendrogram, and scatter plot.

---

## 👨‍💻 Use Case

Agglomerative Clustering is an unsupervised machine learning technique commonly used to group similar data points without prior labels. This technique is applied to the Iris dataset, but it can be adapted for other datasets with numerical features. The results can be used in various fields like customer segmentation, document clustering, and more.
