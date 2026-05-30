# HR Employee Attrition Prediction (Personal Project)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://google.com](https://colab.research.google.com/github/dacamas/HR-Employee-Attrition-Prediction/blob/main/HR_Attrition_Personal_Project.ipynb)

A PyTorch neural network to predict employee attrition using the IBM HR Analytics dataset.

## Project Overview

This project builds a machine learning pipeline to predict whether an employee will leave the company. The model uses a feedforward neural network with batch normalization and early stopping, trained on employee demographics, job characteristics, and satisfaction metrics.

## Dataset

**IBM HR Analytics Employee Attrition Dataset**
- 1,470 employee records
- 34 features (after preprocessing)
- Binary target: Attrition (Yes/No)
- Class distribution: 84% stay, 16% leave

**Features include:**
- Demographics: Age, Gender, MaritalStatus
- Job info: Department, JobRole, JobLevel, OverTime
- Satisfaction scores: JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance
- Tenure: YearsAtCompany, YearsWithCurrManager, YearsSinceLastPromotion
- Compensation: MonthlyIncome, StockOptionLevel, PercentSalaryHike

## Performance Metrics

| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.7724 (test set) |
| **Recall** | 0.6809 (catches 68% of leavers) |
| **Precision** | 0.3721 |
| **F1-Score** | 0.4812 |

**Confusion Matrix:**
```
True Negatives:  193    False Positives: 54
False Negatives:  15    True Positives:  32
```

**Classification Report:**
```
               precision    recall  f1-score   support
          Stay       0.93      0.78      0.85       247
          Leave      0.37      0.68      0.48        47
```

## Steps Taken

### 1. Data Preprocessing
- Removed zero-variance columns (EmployeeNumber, EmployeeCount, StandardHours, Over18)
- Label-encoded categorical features
- Standardized numeric features using StandardScaler
- Identified class imbalance: 84% stay vs 16% leave

### 2. Handling Class Imbalance
- Tested SMOTE and class weights (pos_weight)
- Found pos_weight more effective for neural networks
- Applied pos_weight=5.8 to loss function

### 3. Train/Test Split
- 80/20 split with stratification
- Scaled only numeric features
- Preserved original test set for final evaluation

### 4. Model Architecture

**PyTorch MLP:**
```
Input (34 features)
    ↓
Linear(34, 64) + BatchNorm1d + ReLU + Dropout(0.4)
    ↓
Linear(64, 32) + ReLU + Dropout(0.3)
    ↓
Linear(32, 1) + Sigmoid
```

**Design choices:**
- Batch normalization for training stability
- Dropout for regularization
- Two hidden layers to avoid overfitting on small dataset
- Sigmoid output for binary classification

### 5. Training

**Configuration:**
- Optimizer: Adam (lr=3e-4, weight_decay=1e-3)
- Loss: BCEWithLogitsLoss with pos_weight
- Scheduler: CosineAnnealingLR
- Batch size: 64
- Early stopping: patience=15 epochs

**Training curve:**
```
Epoch  10 | Loss: 1.0802 | Val AUC: 0.7225
Epoch  20 | Loss: 0.9544 | Val AUC: 0.7663
Epoch  30 | Loss: 0.9274 | Val AUC: 0.7864
Epoch  40 | Loss: 0.8524 | Val AUC: 0.7989
Epoch  50 | Loss: 0.8403 | Val AUC: 0.8030
Epoch  60 | Loss: 0.8145 | Val AUC: 0.8044
Epoch  70 | Loss: 0.7702 | Val AUC: 0.8040
Early stopping at epoch 71. Best AUC: 0.8061
```

### 6. Evaluation

- Evaluated on held-out test set (294 samples)
- Computed AUC-ROC, precision, recall, F1-score
- Generated confusion matrix and ROC curve
- Compared against baselines (Logistic Regression, Random Forest)

### 7. Explainability

Used SHAP (SHapley Additive exPlanations) to identify feature importance:

**Top features driving attrition predictions:**
1. OverTime (0.366)
2. EnvironmentSatisfaction (0.300)
3. JobSatisfaction (0.297)
4. NumCompaniesWorked (0.290)
5. YearsSinceLastPromotion (0.285)
6. DistanceFromHome (0.267)

## How to Run

### Install dependencies
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn shap imbalanced-learn
```

### Run the notebook
- Open `hr_attrition.ipynb` in Jupyter or Google Colab
- Run cells sequentially (1-10)
- Dataset loads automatically from GitHub

### File structure
```
├── hr_attrition.ipynb      # Main notebook with all pipeline steps
├── README.md               # This file
├── requirements.txt        # Package dependencies
└── best_model.pt          # Saved PyTorch model weights
```

## Key Results

- **Best validation AUC:** 0.8061 (epoch 71)
- **Test AUC:** 0.7724
- **Recall:** 68% (catches most employees at risk)
- **Correctly identified:** 32 of 47 actual leavers

The model successfully identifies at-risk employees with reasonable recall, capturing the majority of employees likely to leave.
