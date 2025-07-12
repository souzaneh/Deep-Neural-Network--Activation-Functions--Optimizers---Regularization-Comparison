# 🧠 Deep Neural Network on CIFAR-10: Activation Functions, Transfer Learning, Optimizers & Regularization

## 🎯 Project Overview

This project explores deep neural networks (DNN) using the **CIFAR-10** dataset and demonstrates the impact of various techniques on model performance, including activation functions, transfer learning, optimizers, and regularization strategies. The project is implemented in **Python** using **TensorFlow** and **Keras** on **Google Colab**.

---

## 📚 Dataset

- **Dataset:** CIFAR-10 (60,000 32x32 color images in 10 classes)
- **Train/Test Split:** 85% training, 15% testing

---

## ⚙️ Technologies & Libraries

- **Platform:** Python, Google Colab  
- **Frameworks:** TensorFlow, Keras  
- **Libraries:** matplotlib, numpy, pandas, scikit-learn, pickle, time  

---

## 🏗️ Base Model Architecture

- Two dense hidden layers with 50 and 100 neurons  
- Output layer with 10 neurons and softmax activation  
- Optimizer: SGD  
- Loss: categorical_crossentropy  
- Metric: accuracy  

---

## 🔍 Project Tasks & Results

### 1. Activation Functions Comparison

- **SELU Activation** with self-normalizing weight initialization  
- **LeakyReLU Activation** with BatchNormalization  

📊 **Result:**  
SELU activation achieved better accuracy and lower loss on both training and validation datasets compared to LeakyReLU (both traind on 50 epocks).

![Activation Comparison](images/selu_reakurelu_comparison.png)

---

### 2. Transfer Learning on Horse Classification (Binary Classification)

- use the best model that create in previouse step that save in google drive.
 Modified the output to predict whether an image is a horse or not (with 1 output sigmoid activation in last layer and binary classification loss).
- Selected 6,000 images from CIFAR-10 for this task.
- Two scenarios were tested on 10 epocks:
  
  - **Frozen Layers:** 4 layers frozen  
    ⏱️ Training Time: 73.30 seconds  
    🎯 Accuracy: 0.9003
  
  - **Trainable Layers:** all layers trainable  
    ⏱️ Training Time: 91.14 seconds  
    🎯 Accuracy: 0.9031  

✅ Training time increased with more trainable layers, for 10 epocks accuracy showed slight improvement. With more epochs, better accuracy could be achieved.

---

### 3. Optimizer Comparison (50 Epochs)

- **Optimizers Tested:**  
  - SGD  
  - SGD with Momentum  
  - Nesterov SGD with Momentum  
  - AdaGrad  
  - Adam  
  - Nadam  

Accuracy and Loss for each optimizer were plotted and compared.

![Optimizer Comparison](images/Optimizer_Comparison.png)

---
result : According to the graph, it appears that the Adam and Nadam optimizers perform better in reducing loss and increasing accuracy on the training data.
However, by examining the validation loss and accuracy the SGD Optimizers have better performance.

### 4. Regularization Techniques (100 Epochs)

- **Regularization Methods:**  
  - Dropout  
  - Monte Carlo Dropout (MC Dropout)  
  - L1-L2 Regularization  

📊 **Results:**  
- Regularization helped to significantly reduce overfitting (shown by green curves).
- While regularization slightly reduced overall accuracy, it improved generalization performance.
- Applying Monte Carlo Dropout led to slightly better accuracy:  
  - Standard Dropout: 0.4183  
  - MC Dropout: 0.4227  

![Regularization Comparison](images/Regularization.png)

---

## 📝 Code File

- `main_code.ipynb`

---

## 📦 Requirements
 matplotlib
 numpy
 pandas
 scikit-learn
 pickle
 time
 
 
## 👩‍💻 Author

**Souzaneh Sehati**  
GitHub: [https://github.com/souzaneh](https://github.com/souzaneh)