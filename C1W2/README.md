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
├── datasets/
│   └── (cat/non-cat image dataset in .h5 format)
└── README.md                # You're here!
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install numpy matplotlib h5py pillow
```

### 3. Run the code

```bash
python logistic_regression.py
```

You should see training/test accuracy printed in the console and a learning curve plotted with `matplotlib`.

---

## 📈 Sample Output

```
train accuracy: 99.0 %
test accuracy: 70.0 %
```

![learning curve](your_plot_image_if_you_want_to_add_it.png)

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

---

## 🧩 Future Work

- Extend to multi-class classification
- Implement softmax regression
- Apply to real-world datasets
- Add unit tests

---

## 📜 License

This is a learning project based on public educational content.  
Feel free to use or modify for your own educational purposes.
