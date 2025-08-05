# 🧠 Planar Data Classification with One Hidden Layer - Deep Learning Specialization (Week 3)

This repository contains my personal implementation of a simple **two-layer neural network** from scratch using NumPy, inspired by the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) by Andrew Ng.

Original reference notebook:  
[amanchadha's GitHub - Week 3: Planar Data Classification](https://github.com/amanchadha/coursera-deep-learning-specialization/tree/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%203/Planar%20data%20classification%20with%20one%20hidden%20layer)

---

## 📌 Project Overview

This project implements a small neural network (with 1 hidden layer) to classify non-linearly separable 2D data using:

- A hidden layer with **4 neurons**
- **Tanh** and **Sigmoid** activations
- Manual forward and backward propagation
- Cross-entropy cost function
- Gradient descent optimization
- Logistic Regression as a baseline

---

## 🗂️ Folder Structure

```
.
├── planar_classification.py    # Main implementation file
├── planar_utils.py             # Utility functions (provided by course)
├── testCases_v2.py             # Test cases (provided by course)
└── README.md                   # You're here!
```

---

## 📦 Dataset

- The data is synthetically generated using `load_planar_dataset()`
- Each input is a 2D feature (X), with a binary label (Y)
- The dataset is non-linearly separable (ideal for showing the power of neural nets)

---

## 📚 Concepts Practiced

- Forward and backward propagation
- Parameter initialization
- Manual implementation of neural networks
- Binary classification with non-linear decision boundaries
- Model evaluation with accuracy and decision boundary visualization
