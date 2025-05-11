# Disease Prediction using Machine Learning

This project is an independent academic project for predicting 38 disease classes based on clinical symptoms. It explores a complete machine learning pipeline using multiple models (Random Forest, SVM, Naive Bayes) and applies best practices like feature selection, PCA, resampling, calibration, and bootstrap validation.

---

## Dataset

- **Source**: Synthetic medical dataset (included in `dataset/improved_disease_dataset.csv`)
- **Features**: Binary presence/absence of symptoms
- **Target**: Disease (38 unique classes)

---

## ML Pipeline Highlights

- **Data preprocessing** with scaling and encoding
- **Class balancing** using RandomOverSampler and SMOTE
- **Models tested**:  
  - Random Forest 🌲  
  - Support Vector Machine 📐  
  - Naive Bayes 📊  
  - Voting Ensemble 🔗  
- **Evaluation**: Stratified cross-validation, confusion matrices, learning curves
- **Model selection**: GridSearchCV + Validation curves
- **Feature reduction**: PCA + SelectKBest (f-test)
- **Probability calibration**: Confidence vs. Accuracy analysis
- **Uncertainty**: Bootstrap 95% CI for final accuracy

---

## Results

- **Best model**: Random Forest (with PCA + feature selection)
- **Interpretability**: Feature importance and symptom contribution analysis
- **Calibration**: Improved reliability of probability estimates

---

##  Run It Yourself

```bash
pip install -r requirements.txt
python disease_prediction.py
