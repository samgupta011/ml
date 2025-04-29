
# 📊 Standard Deviation and Covariance Analysis on Iris Dataset

This project demonstrates how to compute and interpret **standard deviation** and **covariance** using a built-in dataset: the **Iris dataset**. These statistical tools help us understand **data spread** and **relationships between features**.

---

## 📦 Libraries Used

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
```

- `numpy` and `pandas` for numerical and data frame operations.
- `sklearn.datasets.load_iris` is used to load the Iris dataset.

---

## 🌼 Step 1: Load the Iris Dataset

```python
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
```

- `load_iris()` returns the Iris dataset with features like sepal and petal measurements.
- We convert it into a `pandas.DataFrame` for easier analysis and name the columns appropriately.

---

## 🔍 Step 2: View the Dataset

```python
print("📊 First 5 Rows of the Dataset:\n")
print(df.head())
```

- Displays the first 5 rows to understand what the data looks like.

---

## 📐 Step 3: Compute Standard Deviations

```python
std_devs = df.std()
print("\n📐 Standard Deviations of Each Feature:\n")
print(std_devs)
```

- `.std()` calculates the standard deviation of each numeric feature.
- A high value indicates that the feature's values are more spread out.

---

## 📌 Step 4: Justify the Spread

```python
max_spread_feature = std_devs.idxmax()
max_spread_value = std_devs.max()

print(f"The feature with the largest spread is **'{max_spread_feature}'** with a standard deviation of {max_spread_value:.2f}.")
```

- `idxmax()` returns the column with the highest standard deviation.
- `max()` gives the numeric value of the maximum standard deviation.
- This tells us **which feature varies the most**.

---

## 📈 Step 5: Compute Covariance Matrix

```python
cov_matrix = df.cov()
print("\n📈 Covariance Matrix:\n")
print(cov_matrix)
```

- `.cov()` computes the **covariance matrix**, which shows how features change together.
- Positive covariance → both features increase together.
- Negative covariance → one increases while the other decreases.

---

## 🧪 Sample Output Interpretation

### Standard Deviation Example:
```
sepal length (cm)    0.83
sepal width (cm)     0.43
petal length (cm)    1.76  ← most spread
petal width (cm)     0.76
```

The feature **`petal length (cm)`** has the highest spread (1.76), indicating more variability in this feature.

---

### Covariance Matrix (Example Snippet):

```
                     sepal length  sepal width  ...
sepal length (cm)         0.68569     -0.04243
sepal width (cm)         -0.04243      0.18800
...
```

- `sepal length` and `sepal width` have a small **negative** covariance: as one increases, the other slightly decreases.

---

## ✅ Summary

- **Standard deviation** helps identify which feature has the **most variability**.
- **Covariance** shows **linear relationships** between features.
- This kind of preprocessing and analysis is crucial for understanding feature importance and correlation before applying machine learning.

---

## 🚀 How to Run

1. Install dependencies (if needed):

```bash
pip install numpy pandas scikit-learn
```

2. Run the script in your Python environment (VS Code, Jupyter, etc.)

---

## 📚 Dataset Used

- **Iris Dataset** (from sklearn)
- Features: `sepal length`, `sepal width`, `petal length`, `petal width`
- Classification of Iris flower species

