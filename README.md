# Handwritten-Digits-Classification-MNIST-Dataset-Random_Forest-SVC-KNN-Decision-_Tree
This repository contains code to classify handwritten digits from the MNIST dataset using various machine learning algorithms. The dataset is preprocessed, and different classification models, such as Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and Support Vector Machines, are trained and compared. The performance of each model is evaluated using precision, accuracy, recall, F1 score, and confusion matrices.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Clone repository](#clone-repository)
- [Data Preprocessing](#data-preprocessing)
- [Models Implemented](#models-implemented)
- [Model Comparison Table](#model-comparison-table)
- [License](#license)

---

## Overview
This project aims to classify handwritten digits from the MNIST dataset using a variety of machine learning models. The models are evaluated based on metrics such as precision, accuracy, recall, F1 score, and training time.

---

## Dataset
The MNIST dataset consists of 70,000 28x28 pixel grayscale images of handwritten digits (0-9) and their corresponding labels.

---

## Technologies Used
- Python
- TensorFlow/Keras
- OpenCV
- scikit-learn
- Matplotlib
- NumPy
---

## Clone repository.
   ```bash
   git clone https://github.com/Aziz-ur-Rehman11/Spam-Non-Spam-Email-Text-Classification-SVC-Random-Forest-KNN-K-Clustering.git
   ```

## Data Preprocessing

1. Resizing Images
The images from the MNIST dataset are resized from 28x28 to 10x10 using OpenCV's `resize()` function.

2. Z-Score Normalization
   Z-score normalization is applied to the resized images to standardize the pixel values.

4. Visualizing the Dataset
Random samples from the dataset are visualized before and after resizing and normalization, allowing a quick check of the preprocessed images.

## Models Implemented

### Logistic Regression
- **Algorithm**: Logistic Regression with Stochastic Gradient Descent.
- **Max Iterations**: 500
- **Evaluation Metrics**:
  - Precision: Computed using `precision_score()`
  - Accuracy: Computed using `accuracy_score()`
  - Recall: Computed using `recall_score()`
  - F1 Score: Computed using `f1_score()`
  - Confusion Matrix: Visualized using `ConfusionMatrixDisplay()`

### K-Nearest Neighbors
- **Neighbors**: 5
- **Evaluation Metrics**: Similar to Logistic Regression.

### Decision Tree Classifier
- **Algorithm**: Decision Tree
- **Evaluation Metrics**: Similar to Logistic Regression.

### Random Forest Classifier
- **Algorithm**: Random Forest with 100 trees.
- **Evaluation Metrics**: Similar to Logistic Regression.

### Support Vector Machines
- **Kernel**: Linear
- **Evaluation Metrics**: Similar to Logistic Regression.

---

### Model Comparison Table

| Metric          | SGD-Logistic Regression | K-Nearest Neighbors Classifier | Decision Tree Classifier | Random Forest Classification | Support Vector Machines |
|-----------------|-------------------------|-------------------------------|--------------------------|------------------------------|-------------------------|
| **Precision**    | 0.885968                | 0.942002                      | 0.787386                 | 0.943635                     | 0.924966                |
| **Accuracy**     | 0.885333                | 0.940667                      | 0.786333                 | 0.943333                     | 0.924667                |
| **Recall**       | 0.885333                | 0.940667                      | 0.786333                 | 0.943333                     | 0.924667                |
| **F1 Score**     | 0.885368                | 0.940475                      | 0.786415                 | 0.943376                     | 0.924436                |
| **Training Time**| 12.143859               | 0.003278                      | 2.008909                 | 7.834914                     | 4.272295                |

A bar plot is also generated to compare the performance of each model in terms of precision, accuracy, recall, and F1 score.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

