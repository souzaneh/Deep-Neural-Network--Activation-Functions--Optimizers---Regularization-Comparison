# 🧠 Deep Neural Network on CIFAR-10  
### Activation Functions, Transfer Learning, Optimizers & Regularization

---

## 🎯 Project Overview

This project explores Deep Neural Networks (DNNs) using the **CIFAR-10** dataset and demonstrates the effect of different techniques on model performance, including:

- Activation Functions  
- Transfer Learning  
- Optimizers  
- Regularization Strategies  

The implementation is done in **Python** using **TensorFlow** and **Keras** on **Google Colab**.

---

## 📚 Dataset

- **Name:** CIFAR-10  
- **Description:** 60,000 color images (32x32 pixels) across 10 classes  
- **Split:** 85% training / 15% testing  

---

## ⚙️ Technologies & Libraries

- **Platform:** Python (Google Colab)  
- **Frameworks:** TensorFlow, Keras  
- **Libraries Used:**  
  - `numpy`  
  - `pandas`  
  - `matplotlib`  
  - `scikit-learn`  
  - `pickle`  
  - `time`  

---

## 🏗️ Base Model Architecture

- Dense Layer 1: 50 neurons  
- Dense Layer 2: 100 neurons  
- Output Layer: 10 neurons with `softmax` activation  
- **Optimizer:** SGD  
- **Loss Function:** categorical_crossentropy  
- **Metric:** accuracy  

---

## 🔍 Project Tasks & Results

### 1. 🔌 Activation Functions Comparison

Two types of activation functions were tested:

- **SELU** with self-normalizing weight initialization  
- **LeakyReLU** with BatchNormalization  

**🔍 Result:**  
SELU achieved better accuracy and lower loss on both training and validation datasets over 50 epochs.

![Activation Comparison](images/selu_leakyrelu_comparison.png)

---

### 2. 🐎 Transfer Learning for Horse Classification (Binary Classification)

- Best model from the previous task was reused (loaded from Google Drive).
- Output layer modified for binary classification (`sigmoid`).
- Used a subset of **6,000 CIFAR-10** images.

Two scenarios were evaluated (on 10 epochs):

| Configuration      | Frozen Layers | Accuracy | Training Time |
|--------------------|---------------|----------|---------------|
| Scenario 1         | 4 frozen      | 0.9003   | 73.30s        |
| Scenario 2         | All trainable | 0.9031   | 91.14s        |

✅ Training time increased with more trainable layers. Slight improvement in accuracy suggests better performance could be achieved with more epochs.

---

### 3. ⚙️ Optimizer Comparison (50 Epochs)

**Optimizers Tested:**

- SGD  
- SGD + Momentum  
- Nesterov Momentum  
- AdaGrad  
- Adam  
- Nadam  

**Result Summary:**  
- **Adam** and **Nadam** showed better performance on training loss and accuracy.  
- On validation data, **SGD variants** appeared to generalize better.

![Optimizer Comparison](images/Optimizer_Comparison.png)

---

### 4. 🛡️ Regularization Techniques (100 Epochs)

**Techniques Applied:**

- Dropout  
- Monte Carlo Dropout (MC Dropout)  
- L1-L2 Regularization  

**📈 Results:**

| Method           | Accuracy |
|------------------|----------|
| Dropout          | 0.4183   |
| MC Dropout       | 0.4227   |

✅ Regularization reduced overfitting (green curves).  
While accuracy dropped slightly, generalization improved — MC Dropout gave the best result.

![Regularization Comparison](images/Regularization.png)

---

## 💾 Code

- Notebook: `codes.ipynb`  

---

## 📦 Requirements

```txt
matplotlib  
numpy  
pandas  
scikit-learn  
pickle  
time


👩‍💻 Author
Souzaneh Sehati
GitHub: https://github.com/souzaneh

