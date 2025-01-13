# House Price Prediction

## Introduction

The House Price Prediction project aims to predict housing prices based on various features such as square footage, number of bedrooms, location, etc. The project uses machine learning techniques, particularly deep learning with Keras, to build a predictive model. This project is part of an effort to explore data preprocessing, feature engineering, model training, and evaluation in a real-world context.

## Objective

The primary objective of this project is to:
1. Preprocess the data for use in machine learning models.
2. Train a model to predict house prices.
3. Evaluate the model’s performance using appropriate metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE).
4. Fine-tune the model to improve predictions.

## Steps and Files Worked With

### 1. Data Preprocessing
   - **File:** `src/data_preprocessing.py`
   - **Description:** This script loads the dataset, performs normalization (scaling the data), and splits it into training and test sets. It ensures the data is clean and ready for model training.

### 2. Model Training and Evaluation
   - **File:** `src/evaluate.py`
   - **Description:** This script handles the training of the model using Keras, implements cross-validation (processing different folds), and evaluates model performance using MSE and MAE. It also saves the trained model for later use.

### 3. Results and Evaluation
   - **Metrics Used:**
     - Mean Squared Error (MSE)
     - Mean Absolute Error (MAE)
   - **Model Output:** The model achieved a test MSE of 17.89 and a test MAE of 2.69 after processing multiple folds of the dataset.

### 4. Data
   - **Dataset:** A modified version of the Boston Housing dataset (or another relevant dataset for house price prediction).
   - **Features:** Includes various features like the number of rooms, crime rate, property tax rates, etc.

## Challenges

- **Data Quality:** The dataset had some missing values that needed to be addressed during preprocessing.
- **Model Overfitting:** Initially, the model showed signs of overfitting, where it performed well on training data but poorly on test data. This was mitigated through techniques like regularization and tweaking hyperparameters.
- **Cross-Validation:** Ensuring proper implementation of cross-validation was a challenge but critical for evaluating the model performance thoroughly.
  
## Results

- The model’s performance metrics after training on multiple folds:
  - **Test MSE:** 17.89
  - **Test MAE:** 2.69
- **Training Loss:** 13.31
- **Training MAE:** 2.51

While the results indicate reasonable performance, there is still room for improvement in terms of accuracy, which can be achieved through hyperparameter tuning, using more advanced models, or refining the data preprocessing steps.

## Achievements

- Successfully implemented data preprocessing techniques such as normalization.
- Built and trained a deep learning model using Keras to predict house prices.
- Evaluated the model's performance using appropriate metrics (MSE and MAE).
- Addressed model overfitting issues and improved generalization through cross-validation.

## Takeaways

- **Data Preprocessing is Crucial:** Proper data handling and normalization play a key role in achieving good model performance.
- **Model Evaluation:** Cross-validation helps ensure the model’s robustness and generalizability.
- **Continuous Improvement:** There’s always room to enhance model performance through hyperparameter optimization and feature engineering.

## Future Work

- Try other machine learning algorithms like Random Forest, Gradient Boosting, and XGBoost to compare their performance with deep learning models.
- Further tune the hyperparameters of the deep learning model for improved results.
- Collect more data or add additional features to improve model accuracy.

## Files Overview

- `src/data_preprocessing.py`: Data preprocessing script.
- `src/evaluate.py`: Model training and evaluation script.
- `src/`: Directory containing other utility scripts (if applicable).
- `notebooks/`: Jupyter notebooks for exploring the dataset and results (if any).
- `requirements.txt`: Lists all dependencies required to run the project.
- `README.md`: This file, outlining the details of the project.

