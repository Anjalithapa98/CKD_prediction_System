Chronic Kidney Disease Classification Using Machine Learning

A comparative machine learning project that evaluates Random Forest, Logistic Regression, Support Vector Machine (SVM), and Naïve Bayes for the early detection of Chronic Kidney Disease (CKD) using clinical and laboratory data from the UCI Machine Learning Repository. The project applies comprehensive data preprocessing, exploratory data analysis (EDA), feature engineering, hyperparameter tuning, and model evaluation to determine the most accurate and reliable classifier for CKD prediction.

The developed system assists in identifying patients at risk of Chronic Kidney Disease before the disease progresses to advanced stages. By leveraging machine learning techniques, the project aims to support healthcare professionals in making faster and more informed clinical decisions while demonstrating the potential of AI in medical diagnosis.

This project was completed as the Major Project for the Bachelor of Computer Engineering degree at Everest Engineering College, Lalitpur, Nepal (January 2026) under the supervision of Er. Narayan Sapkota by;
Anjali Thapa,
Nirmal Khatri,
Prisma Sharma,
Pujan Burlakoti,

Overview

Chronic Kidney Disease (CKD) is a progressive medical condition that often remains undiagnosed until significant kidney damage has occurred. Early diagnosis is essential for reducing complications and improving patient outcomes. Traditional diagnostic procedures can be expensive, time-consuming, and inaccessible in resource-limited settings.

This project approaches CKD prediction as a binary classification problem, where patient clinical and laboratory measurements are analyzed to determine whether a patient is likely to have CKD. Multiple machine learning algorithms are trained, optimized, and compared under identical preprocessing and evaluation conditions to identify the most effective predictive model.

The project includes:

Comprehensive data cleaning and missing value imputation
Exploratory Data Analysis (Histogram, Box Plot, and Violin Plot)
Feature preprocessing and encoding
Hyperparameter tuning
Comparative analysis of four supervised learning algorithms
Performance evaluation using multiple statistical metrics
ROC-AUC analysis and confusion matrix visualization
Web-based prediction interface for CKD classification
Models Evaluated

Four machine learning algorithms were implemented and benchmarked using the same dataset and evaluation strategy:

Random Forest — An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.
Logistic Regression — A robust baseline classifier widely used in medical diagnosis for binary classification tasks.
Support Vector Machine (SVM) — A margin-based classifier capable of effectively separating CKD and non-CKD patient records.
Naïve Bayes — A probabilistic classifier that offers fast training and efficient prediction on structured medical datasets.
