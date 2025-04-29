
# 🧼 Data Preprocessing & Analysis on Titanic Dataset

This project demonstrates how to perform data preprocessing using a pre-built dataset — the **Titanic dataset** — available from the `seaborn` library. It covers:
- Basic dataset analysis
- Handling missing values
- Understanding and cleaning dirty data
- Preprocessing continuous variables (scaling and binning)

---

## 📦 Libraries Used

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
```

We import:
- **pandas** and **numpy** for data manipulation
- **seaborn** for loading the Titanic dataset
- **sklearn.preprocessing** tools for normalization and binning

---

## 🔍 Step 1: Load and Analyze the Data

```python
df = sns.load_dataset('titanic')
```

We load the Titanic dataset which contains details about passengers like age, sex, ticket class, survival status, etc.

```python
print(df.info())
print(df.head())
print(df.describe(include='all'))
```

- `.info()` gives column types and missing values
- `.head()` shows the first 5 rows
- `.describe()` gives statistics like mean, std, min, max for numeric and categorical columns

---

## 🧹 Step 2: What is Dirty Data?

We explain what dirty data is:
- Missing values
- Inconsistent data types
- Duplicates
- Outliers
- Incorrect formats

Handling dirty data is crucial before building models or doing analysis.

---

## ❌ Step 3: Handle Null (Missing) Values

```python
print(df.isnull().sum())
```

This line counts the number of missing values in each column.

```python
df_cleaned = df.drop(columns=['deck'])
```

We drop the `deck` column because it has too many missing values.

```python
df_cleaned['age'].fillna(df['age'].median(), inplace=True)
df_cleaned['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
```

- Missing `age` is filled with the **median**
- Missing `embarked` is filled with the **mode** (most frequent value)

```python
df_cleaned.dropna(inplace=True)
```

Finally, any remaining rows with null values are removed.

---

## 📏 Step 4: Handle Continuous Values

### 🧮 Normalize Age and Fare

```python
scaler = MinMaxScaler()
df_cleaned[['age_norm', 'fare_norm']] = scaler.fit_transform(df_cleaned[['age', 'fare']])
```

We use `MinMaxScaler` to scale age and fare values between **0 and 1**.

### 📊 Bin (Discretize) Age into Categories

```python
kbin = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
df_cleaned['age_binned'] = kbin.fit_transform(df_cleaned[['age']])
```

We convert continuous age into 4 equal-width **bins** using `KBinsDiscretizer`. This is useful when you want to convert a continuous variable into a categorical one.

---

## ✅ Final Output

```python
print(df_cleaned.head())
```

Displays the cleaned and preprocessed dataset, which now has:
- No missing values
- Normalized continuous values
- Binned age column

---

## 📌 Key Concepts Summary

- **Dirty Data** must be cleaned before model training
- **Missing Values** can be filled or dropped depending on context
- **Normalization** puts values in the same scale (0–1)
- **Binning** transforms continuous values into categorical bins

---

## 🚀 To Run This Code

Make sure you have the required libraries:

```bash
pip install pandas numpy seaborn scikit-learn
```

Then run the Python script in any environment (like Jupyter Notebook, VS Code, or command line).

---

## 📂 Dataset Source

Titanic dataset is available in `seaborn` as a built-in example dataset:
```python
df = sns.load_dataset('titanic')
```

