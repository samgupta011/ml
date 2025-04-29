
# McCulloch-Pitts Model for Logical Gates

This project demonstrates the implementation of the **McCulloch-Pitts model**, a simple mathematical model for a neuron. The model can simulate basic **logical gates** such as **AND**, **OR**, and **NOT** using a threshold-based activation function.

---

## 📦 Requirements

You can install the necessary libraries using `pip`:

```bash
pip install numpy
```

---

## 🧠 Understanding the McCulloch-Pitts Model

The McCulloch-Pitts model is a binary model that computes an output based on a weighted sum of inputs and compares it to a threshold. If the weighted sum exceeds the threshold, the output is activated (set to 1), otherwise, the output is 0.

This model simulates basic logic gates like AND, OR, and NOT:

- **AND Gate**: The output is 1 only when both inputs are 1.
- **OR Gate**: The output is 1 when at least one of the inputs is 1.
- **NOT Gate**: The output is the opposite of the input.

---

## 📁 Code Explanation

### 1. **Import Libraries**

```python
import numpy as np
```

- `numpy` is imported for array and matrix operations, as the logic gates rely on weighted sums, which are calculated using `numpy`.

---

### 2. **Activation Function**

```python
def activation_function(x):
    return 1 if x >= 1 else 0
```

- The `activation_function(x)` implements a **step function**: if the input `x` is greater than or equal to `1`, the output is `1`; otherwise, it's `0`. This mimics the threshold-based activation of a neuron.

---

### 3. **McCulloch-Pitts Model for AND Gate**

```python
def AND_gate(x1, x2):
    weights = np.array([1, 1])  # weights for inputs x1 and x2
    threshold = 2  # Threshold for AND gate
    input_vector = np.array([x1, x2])
    
    # Calculate weighted sum
    weighted_sum = np.dot(input_vector, weights)
    
    # Apply activation function
    output = activation_function(weighted_sum - threshold)
    return output
```

- The AND gate has two inputs (`x1`, `x2`), each with a weight of `1`. The threshold is set to `2`. The weighted sum of inputs is calculated and passed through the activation function.

---

### 4. **McCulloch-Pitts Model for OR Gate**

```python
def OR_gate(x1, x2):
    weights = np.array([1, 1])  # weights for inputs x1 and x2
    threshold = 1  # Threshold for OR gate
    input_vector = np.array([x1, x2])
    
    # Calculate weighted sum
    weighted_sum = np.dot(input_vector, weights)
    
    # Apply activation function
    output = activation_function(weighted_sum - threshold)
    return output
```

- Similar to the AND gate, but the threshold is set to `1` for the OR gate. The output will be `1` if at least one of the inputs is `1`.

---

### 5. **McCulloch-Pitts Model for NOT Gate**

```python
def NOT_gate(x):
    weights = np.array([-1])  # Negative weight for NOT gate
    threshold = 0  # Threshold for NOT gate
    input_vector = np.array([x])
    
    # Calculate weighted sum
    weighted_sum = np.dot(input_vector, weights)
    
    # Apply activation function
    output = activation_function(weighted_sum - threshold)
    return output
```

- The NOT gate is a single-input gate with a negative weight (`-1`). If the input is `0`, the output is `1`, and if the input is `1`, the output is `0`.

---

### 6. **Testing the Gates**

```python
# Test AND gate
print("AND Gate Results:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"AND({x1}, {x2}) = {AND_gate(x1, x2)}")

# Test OR gate
print("\nOR Gate Results:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"OR({x1}, {x2}) = {OR_gate(x1, x2)}")

# Test NOT gate
print("\nNOT Gate Results:")
for x in [0, 1]:
    print(f"NOT({x}) = {NOT_gate(x)}")
```

- We test all possible combinations of inputs for the AND, OR, and NOT gates. The results are printed for each combination.

---

## 📊 Output

The program will display the following outputs for each gate:

```text
AND Gate Results:
AND(0, 0) = 0
AND(0, 1) = 0
AND(1, 0) = 0
AND(1, 1) = 1

OR Gate Results:
OR(0, 0) = 0
OR(0, 1) = 1
OR(1, 0) = 1
OR(1, 1) = 1

NOT Gate Results:
NOT(0) = 1
NOT(1) = 0
```

---

## 💡 Key Takeaways

- The **McCulloch-Pitts model** is a simple model for simulating neural functions and logical operations based on weighted sums and thresholds.
- This program demonstrates how the McCulloch-Pitts model can be used to simulate basic logical gates (AND, OR, and NOT).
- The logic gates are modeled using an **activation function** (step function) that produces a binary output based on the weighted sum of inputs.

---

## 🚀 How to Run

1. Install the required dependency (`numpy`).
2. Copy the code into a Python file or a Jupyter notebook.
3. Run the script, and it will display the output of the logical gates for all possible inputs.

---

## 👨‍💻 Use Case

This model is foundational for understanding how simple artificial neurons work. It's a precursor to more complex neural networks and can be used for simulating basic logic operations in an artificial neural network.
