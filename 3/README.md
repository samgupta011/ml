
# 🏡 Linear Regression on California Housing Dataset

This project demonstrates how to apply **Linear Regression** to a real-world dataset using **scikit-learn**. The dataset contains housing data from California, such as average rooms, location data, and income, with the goal of predicting **median house prices**.

---

## 📦 Libraries Required

```bash
pip install pandas numpy scikit-learn matplotlib
```

These libraries help in data loading, processing, modeling, and visualization.

---

## 📁 Step-by-Step Explanation of the Code

### 1. **Import Libraries**
```python
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
```
- Imports all necessary libraries:
  - `pandas`, `numpy` for data manipulation
  - `sklearn` for dataset, model, splitting, and evaluation
  - `matplotlib` for plotting

---

### 2. **Load Dataset**
```python
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['Target'] = california.target
```
- Loads the **California Housing** dataset
- Converts it to a `DataFrame` for easier handling
- Adds the target variable (house prices)

---

### 3. **View Dataset**
```python
print("🔍 Dataset Preview:")
print(df.head())
print("\n📊 Dataset Summary:")
print(df.describe())
```
- Prints the first 5 rows and a statistical summary (mean, std, min, etc.)

---

### 4. **Define Features and Target**
```python
X = df.drop('Target', axis=1)
y = df['Target']
```
- `X` contains all independent variables (features)
- `y` is the dependent variable (house value)

---

### 5. **Split the Data**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Splits data into **training (80%)** and **testing (20%)**
- `random_state=42` ensures reproducibility

---

### 6. **Train the Linear Regression Model**
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
- Initializes and trains a **Linear Regression** model on the training data

---

### 7. **Predict Using Test Set**
```python
y_pred = model.predict(X_test)
```
- Predicts target values for the test dataset

---

### 8. **Evaluate the Model**
```python
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n📈 R² Score: {r2:.4f}")
print(f"🧮 Mean Squared Error: {mse:.4f}")
print(f"📉 Root Mean Squared Error: {rmse:.4f}")
```
- **R² Score**: How well the model explains the variance in the target
- **MSE**: Average squared error between actual and predicted values
- **RMSE**: Square root of MSE (same unit as target)

---

### 9. **Plot Actual vs Predicted Values**
```python
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.grid(True)
plt.show()
```
- Creates a scatter plot to compare predicted vs actual values
- Ideal prediction would fall on the red dashed diagonal line

---

## 📌 Summary

- This project shows how to train a **simple linear regression model** on real-world housing data.
- Evaluation metrics like R², MSE, and RMSE provide insights into model performance.
- Visualization gives an intuitive understanding of prediction quality.

---

## 🚀 To Run

1. Clone or download this repository
2. Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

3. Run the script in any Python IDE or Jupyter Notebook.

---

## 📚 Dataset Info

- Dataset: California Housing (from sklearn)
- Target: Median House Value
- Features: Income, rooms, bedrooms, population, location, etc.

---
