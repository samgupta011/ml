
# Support Vector Machine (SVM) - Iris Dataset

This project demonstrates the use of **Support Vector Machine (SVM)**, a supervised learning algorithm, to classify the **Iris dataset**. The goal is to classify the Iris flowers into different species based on their sepal and petal measurements.

---

## 📦 Requirements

To run the code, install the following Python libraries using the `pip` command:

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Libraries:
- **`numpy`**: Used for numerical operations.
- **`pandas`**: For data manipulation and handling datasets.
- **`scikit-learn`**: Machine learning library that includes the SVM classifier and tools for model evaluation.
- **`matplotlib`**: For visualizing the data and results.

---

## 🧠 Overview of SVM (Support Vector Machine)

Support Vector Machine (SVM) is a supervised machine learning algorithm commonly used for classification tasks. The algorithm tries to find a hyperplane that best separates the data points of different classes. SVM is especially effective in high-dimensional spaces, making it a powerful method for complex classification problems.

In this project, we apply SVM to the **Iris dataset**, which consists of 150 samples of Iris flowers from three species: **Setosa**, **Versicolor**, and **Virginica**. The features are sepal length, sepal width, petal length, and petal width.

---

## 🖥️ Code Explanation

### 1. **Import Libraries**

```python
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
```

- **`numpy`**: Provides mathematical operations on arrays.
- **`pandas`**: Used to load and manipulate the dataset.
- **`datasets`**: Loads built-in datasets (like Iris).
- **`SVC`**: Implements the Support Vector Classifier for classification tasks.
- **`train_test_split`**: Used to split the dataset into training and testing sets.
- **`classification_report`, `accuracy_score`**: Used to evaluate the performance of the model.
- **`matplotlib`**: For visualizing results.

### 2. **Load the Dataset**

```python
iris = datasets.load_iris()
```

- Loads the **Iris dataset** containing features such as sepal length, sepal width, petal length, and petal width.

### 3. **Prepare the Data**

```python
X = iris.data
y = iris.target
```

- **`X`**: The features of the dataset (sepal length, sepal width, petal length, and petal width).
- **`y`**: The target labels (species of Iris flowers).

### 4. **Split the Data into Training and Testing Sets**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- Splits the dataset into training (80%) and testing (20%) sets to evaluate the performance of the model.

### 5. **Create and Train the SVM Model**

```python
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
```

- **`SVC(kernel='linear')`**: Creates a linear Support Vector Classifier model. The kernel is set to 'linear' for a linear decision boundary.
- **`fit()`**: Trains the SVM model using the training data.

### 6. **Make Predictions**

```python
y_pred = svm_model.predict(X_test)
```

- **`predict()`**: Makes predictions on the test set based on the trained model.

### 7. **Evaluate the Model**

```python
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

- **`accuracy_score()`**: Prints the accuracy of the model.
- **`classification_report()`**: Provides a detailed classification report, including precision, recall, and F1-score for each class.

### 8. **Visualize the Decision Boundary**

```python
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', marker='o', edgecolor='k', alpha=0.7)
plt.title("SVM Classifier - Iris Dataset")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
```

- Visualizes the results of the SVM classifier on the test set using the first two features of the Iris dataset (sepal length and sepal width).

---

## 📊 Expected Output

- **Accuracy**: The accuracy of the SVM classifier on the test set.
- **Classification Report**: A report containing precision, recall, F1-score, and support for each class.
- **Decision Boundary Plot**: A scatter plot showing the classification results of the SVM model.

---

## 💡 Key Takeaways

- **SVM** is a powerful supervised learning algorithm that performs well on high-dimensional datasets.
- The **linear kernel** is used here, but SVM can also be applied with non-linear kernels (e.g., RBF).
- **Evaluation metrics** such as accuracy, precision, recall, and F1-score provide a good understanding of the model's performance.
- **Visualizations** help to understand how the classifier is performing on the dataset.

---

## 🚀 How to Run

1. Install the required libraries using the `pip` command mentioned above.
2. Copy the provided code into a Python file or Jupyter notebook.
3. Run the script to view the classifier's performance and decision boundary plot.

---

## 👨‍💻 Use Case

Support Vector Machines are widely used for classification tasks in various fields such as image recognition, spam detection, and medical diagnosis. By applying SVM to the Iris dataset, we can classify flower species based on their physical features. This approach can be adapted for other classification problems with numerical or categorical data.
