
# 🎯 Logistic Regression with Breast Cancer Dataset

This project demonstrates how to apply **Logistic Regression** using the built-in **Breast Cancer Wisconsin dataset** from `scikit-learn`. The objective is to classify tumors as **benign** or **malignant** based on 30 features.

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

---

## 📁 Line-by-Line Code Explanation

### 1. **Import Libraries**

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
```

- `pandas`, `numpy` → data handling
- `load_breast_cancer` → loads the dataset
- `train_test_split` → splits dataset into train/test
- `LogisticRegression` → model
- `accuracy_score`, `confusion_matrix`, `classification_report` → evaluation
- `seaborn`, `matplotlib.pyplot` → plotting

---

### 2. **Load the Dataset**

```python
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target
```

- Loads breast cancer data and converts it to a `pandas DataFrame`
- Adds the target column (0 = malignant, 1 = benign)

---

### 3. **View Data**

```python
print("🔍 Dataset Head:")
print(df.head())
print("\n📊 Dataset Description:")
print(df.describe())
print("\n🎯 Target Classes:", np.unique(df['target']))
```

- Shows the first 5 rows
- Describes statistics (mean, std, min, max)
- Lists unique values in the target

---

### 4. **Split into Features & Target**

```python
X = df.drop('target', axis=1)
y = df['target']
```

- `X` → all input features
- `y` → output labels (classification targets)

---

### 5. **Train-Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- Splits data: 80% training, 20% testing
- `random_state=42` makes results reproducible

---

### 6. **Train the Logistic Regression Model**

```python
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
```

- Initializes Logistic Regression
- `max_iter=10000` prevents convergence warnings
- Fits the model to training data

---

### 7. **Make Predictions**

```python
y_pred = model.predict(X_test)
```

- Predicts labels for test data

---

### 8. **Evaluate the Model**

```python
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print(f"\n✅ Accuracy: {acc:.4f}")
print("\n📉 Confusion Matrix:")
print(cm)
print("\n📋 Classification Report:")
print(cr)
```

- `Accuracy` → how often predictions are correct
- `Confusion Matrix` → how many correct/incorrect per class
- `Classification Report` → precision, recall, f1-score

---

### 9. **Visualize Confusion Matrix**

```python
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

- Uses a heatmap to visualize classification accuracy
- Reduces complexity when interpreting true/false positives/negatives

---

## 📌 Summary

- **Model Used**: Logistic Regression
- **Dataset**: Breast Cancer Wisconsin Dataset (binary classification)
- **Evaluation**: Accuracy, Confusion Matrix, Classification Report
- **Visualization**: Confusion matrix heatmap with `seaborn`

---

## 🚀 How to Run

1. Install dependencies  
2. Copy the code into a Python file or Jupyter Notebook  
3. Run the script to see predictions, accuracy, and confusion matrix

---

## 📚 Target Class Mapping

- `0` → Malignant (cancerous)
- `1` → Benign (non-cancerous)

---

## 👩‍⚕️ Use Case

Early detection and classification of tumors can help doctors make faster and more accurate diagnoses. Logistic Regression offers a lightweight and interpretable model suitable for such classification tasks.

