# Loan Prediction Analysis Project

This project analyzes loan approval patterns using machine learning techniques to predict whether a loan application will be approved or rejected.

## 📁 Project Structure

```
project2/
├── loan_prediction_analysis.ipynb    # Main analysis notebook (English version)
├── train.csv                        # Training dataset (615 records)
├── test.csv                         # Test dataset (368 records)
└── README.md                        # This file
```

## 🎯 Project Overview

**Business Problem**: Dream Housing Finance company wants to automate their loan eligibility process based on customer details provided in online application forms.

**Technical Approach**: Binary classification problem to predict loan approval status (Y/N).

## 📊 Dataset Information

- **Training Data**: 615 loan applications with approval status
- **Test Data**: 368 loan applications for prediction
- **Features**: Gender, Marital Status, Education, Income, Loan Amount, Credit History, Property Area, etc.
- **Target**: Loan_Status (Y=Approved, N=Rejected)

## 🚀 How to Run

1. **Prerequisites**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook loan_prediction_analysis.ipynb
   ```

3. **Run Cells Sequentially**: Execute each cell in order to complete the full analysis pipeline.

## 📝 Analysis Pipeline

The notebook follows a comprehensive data science workflow:

1. **Environment Setup** - Import libraries and configure settings
2. **Data Loading** - Load training and test datasets
3. **Initial Data Exploration** - Understand data structure and characteristics
4. **Missing Values Analysis** - Identify and plan missing data treatment
5. **Target Variable Analysis** - Examine loan approval distribution
6. **Feature Distribution Analysis** - Explore categorical and numerical features
7. **Feature-Target Relationship Analysis** - Understand predictive relationships
8. **Data Preprocessing** - Clean and prepare data for modeling
9. **Feature Engineering** - Create new meaningful features
10. **Model Development** - Build and compare multiple ML models
11. **Model Evaluation** - Detailed performance analysis
12. **Final Predictions** - Generate predictions for test data
13. **Project Summary** - Key findings and conclusions

## 🤖 Machine Learning Models

The analysis compares multiple algorithms:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

## 📈 Expected Outputs

When you run the notebook, it will generate:
- **loan_prediction_submission.csv**: Final predictions for competition submission
- **detailed_predictions.csv**: Predictions with probability scores
- **train_processed.csv**: Cleaned training data
- **test_processed.csv**: Cleaned test data

## 🎓 Learning Objectives

This project demonstrates:
- Complete machine learning pipeline implementation
- Data preprocessing and feature engineering techniques
- Model comparison and evaluation strategies
- Business problem solving with data science
- Professional code documentation and visualization

## 📋 Requirements

- Python 3.7+
- Jupyter Notebook
- pandas, numpy, matplotlib, seaborn, scikit-learn

## 🏆 Results

The project achieves good predictive performance through:
- Comprehensive exploratory data analysis
- Thoughtful feature engineering
- Multiple model comparison
- Robust evaluation methodology

---

**Happy Learning!** 🎉

*This project is based on the Analytics Vidhya Loan Prediction dataset and serves as an excellent introduction to machine learning classification problems.* 