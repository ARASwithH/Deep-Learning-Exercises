# 🧠 Neural Network from Scratch (4-Layer) - Deep Learning Specialization (Week 4)

This repository contains my personal implementation of a **4-layer deep neural network** from scratch using NumPy, inspired by the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) by Andrew Ng.

Original reference notebook:  
[amanchadha's GitHub - Week 4: Deep Neural Network](https://github.com/amanchadha/coursera-deep-learning-specialization/tree/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%204)

---

## 📌 Project Overview

This project implements a **binary image classifier** (cat vs. non-cat) using a fully connected neural network with **4 layers**, covering:

- Data loading and preprocessing  
- Parameter initialization for deep networks  
- Forward propagation through multiple layers  
- Computation of cost function (cross-entropy)  
- Backward propagation  
- Gradient descent optimization  
- Predictions on test and training sets  
- Visualization of learning curve

---

## 🗂️ Folder Structure

```bash
.
├── dnn_app.py               # Main implementation file (4-layer NN)
├── dnn_utils.py             # Helper functions for forward/backward propagation
├── test_catvnoncat.h5       # Test dataset
├── train_catvnoncat.h5      # Training dataset
└── README.md                # You're here!
```

---

## 📦 Dataset

The dataset is provided in `.h5` format and contains:

- **Training set** of labeled cat/non-cat images  
- **Test set** with the same structure  

Each image has dimensions `(num_px, num_px, 3)` and is flattened into vectors of shape `(num_px * num_px * 3, 1)` before being fed into the network.

---

## 📚 Concepts Practiced

- Deep neural networks (4-layer architecture)  
- Parameter initialization strategies  
- Forward and backward propagation for multi-layer networks  
- Binary cross-entropy loss  
- Gradient descent optimization  
- Use of NumPy for vectorized computations  
- Building a full training pipeline from scratch  
