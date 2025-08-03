# 🧠 Logistic Regression from Scratch - Deep Learning Specialization (Week 2)

This repository contains my personal implementation of a **logistic regression model** from scratch using NumPy, inspired by the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) by Andrew Ng.

Original reference notebook:  
[amanchadha's GitHub - Week 2: Logistic Regression](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%202/Logistic%20Regression%20as%20a%20Neural%20Network/Logistic.ipynb)

---

## 📌 Project Overview

This project implements a simple **binary classifier** (cat vs. non-cat) using logistic regression, including:

- Data loading and preprocessing  
- Parameter initialization  
- Forward and backward propagation  
- Gradient descent optimization  
- Predictions on test and training sets  
- Visualization of learning curve

---

## 🗂️ Folder Structure

```
.
├── logistic_regression.py   # Main implementation file
├── lr_utils.py              # Utility functions for loading dataset
├── test_catvnoncat.h5       # dataset
├── train_catvnoncat.h5      # dataset
└── README.md                # You're here!
```

---

## 📦 Dataset

The dataset is loaded via `lr_utils.py` and contains:

- Training set of cat/non-cat images in `.h5` format
- Test set with the same structure

Each image is of shape `(num_px, num_px, 3)` and flattened to `(num_px * num_px * 3, 1)` for training.

---

## 📚 Concepts Practiced

- Logistic regression
- Vectorization with NumPy
- Binary cross-entropy cost
- Manual implementation of gradient descent
- Image preprocessing and flattening


