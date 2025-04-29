
# Naive Bayes Classifier - Iris Dataset

This project demonstrates the implementation of the **Naive Bayes classifier** using the **Iris dataset**. The Naive Bayes model is a probabilistic classifier based on Bayes' Theorem, assuming independence between features. This implementation uses the **Gaussian Naive Bayes** model, which is suited for continuous data.

---

## 📦 Requirements

You can install the necessary dependencies using `pip`:

```bash
pip install numpy pandas scikit-learn
```

---

## 🧠 Overview of Naive Bayes Classifier

The **Naive Bayes classifier** is a simple probabilistic classifier based on the **Bayes' Theorem** with an assumption of independence between the features. The model computes the probability of each class given the features and predicts the class with the highest probability.

In this implementation, we are using the **Gaussian Naive Bayes** model, which is useful when the features are continuous and follow a normal distribution.

---

## 📁 Code Explanation

### 1. **Importing Libraries**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

- We import `numpy` and `pandas` for data manipulation and handling.
- `train_test_split` from `sklearn.model_selection` is used to split the dataset into training and testing sets.
- `GaussianNB` from `sklearn.naive_bayes` implements the Naive Bayes classifier for continuous data.
- `datasets` from `sklearn` provides built-in datasets, such as the Iris dataset.
- `accuracy_score`, `confusion_matrix`, and `classification_report` are used to evaluate the performance of the model.

---

### 2. **Loading the Dataset**

```python
iris = datasets.load_iris()
```

- The **Iris dataset** is loaded using `datasets.load_iris()`, which contains data about Iris flowers, including four features (sepal length, sepal width, petal length, and petal width) and a target variable (species).

---

### 3. **Creating DataFrame**

```python
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
```

- A **pandas DataFrame** is created from the Iris dataset, with the feature names as column names. The target class (`species`) is added as a new column.

---

### 4. **Splitting the Dataset**

```python
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

- We separate the features (`X`) and the target variable (`y`).
- The dataset is split into training (70%) and testing (30%) sets using `train_test_split`.

---

### 5. **Training the Naive Bayes Model**

```python
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)
```

- We initialize a **Gaussian Naive Bayes** classifier (`GaussianNB`).
- The model is trained on the training data (`X_train`, `y_train`).

---

### 6. **Making Predictions**

```python
y_pred = naive_bayes_classifier.predict(X_test)
```

- The trained model is used to make predictions on the test data (`X_test`).

---

### 7. **Evaluating the Model**

```python
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
```

- The **accuracy score** of the model is calculated using `accuracy_score`.
- The **confusion matrix** is generated to show how well the model performed in each class.
- The **classification report** provides detailed metrics like precision, recall, and F1-score for each class.

---

### 8. **Displaying the Results**

```python
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
```

- The results, including the accuracy, confusion matrix, and classification report, are printed.

---

## 📊 Output

The program will display the following outputs for the Iris dataset:

```text
Accuracy: 97.78%

Confusion Matrix:
[[14  0  0]
 [ 0 15  1]
 [ 0  0 15]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        14
           1       1.00      0.94      0.97        16
           2       0.94      1.00      0.97        15

    accuracy                           0.98        45
   macro avg       0.98      0.98      0.98        45
weighted avg       0.98      0.98      0.98        45
```

---

## 💡 Key Takeaways

- The **Naive Bayes classifier** is a powerful and simple probabilistic model that assumes conditional independence between features.
- This implementation uses the **Gaussian Naive Bayes** model, which is ideal for continuous data that is normally distributed.
- The **Iris dataset** is used here to demonstrate the classification of different species of Iris flowers based on their features.

---

## 🚀 How to Run

1. Install the required dependencies (`numpy`, `pandas`, and `scikit-learn`).
2. Copy the code into a Python file or a Jupyter notebook.
3. Run the script, and it will display the output of the Naive Bayes classifier's accuracy, confusion matrix, and classification report.

---

## 👨‍💻 Use Case

This project is useful for understanding how the Naive Bayes classifier works for classification problems. It can be extended to work on other datasets, and the principles demonstrated here can be applied to more complex problems in data science and machine learning.

