# Binary Income Classification using Various ML Algorithms(UCI Adult Dataset)

## Overview
This project focuses on predicting whether an individual earns more than \$50K per year using the UCI Adult Income dataset. The task is formulated as a binary classification problem using demographic and employment-related features.

---

## Dataset
- Source: UCI Machine Learning Repository (Adult Dataset)
- Features include:
  - Age, education, occupation, hours-per-week, etc.
- Target:
  - Income >50K or ≤50K

---

## Project Pipeline

### 1. Data Cleaning
- Fixed inconsistencies between train and test labels (e.g., `>50K` vs `>50K.`)
- Handled missing values
- Removed duplicates

### 2. Exploratory Data Analysis (EDA)
- Statistical summaries (`describe`, `info`)
- Distribution analysis of key features
- Class distribution analysis

### 3. Preprocessing
- Encoding categorical variables
- Feature scaling for numerical features
- Combined train/test preprocessing for consistency

### 4. Handling Class Imbalance
- Applied resampling techniques to balance the dataset

---

## Models Implemented

The following models were trained and evaluated:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Support Vector Machine (Linear & RBF kernels)  
- Multi-Layer Perceptron (Neural Network)  
- XGBoost
- LightGBM  

Hyperparameter tuning was performed using Grid Search / Random Search for selected models.

---

## Results
- Compared multiple models based on accuracy and performance metrics  
- Ensemble methods showed stronger performance than other methods 
- Tree-based methods and MLP had promising results as well
- The best models were **XGBoost**, **LightGBM** achieving an AUC of roughly 0.8
  
---

## Technologies Used
- Python  
- NumPy, Pandas  
- scikit-learn  
- Matplotlib
- Seaborn  
- XGBoost  

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
