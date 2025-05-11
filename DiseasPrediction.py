# %% 
# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode

# Sklearn imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV, 
                                   learning_curve, validation_curve, KFold, StratifiedKFold)
from sklearn.metrics import (accuracy_score, roc_curve, auc, precision_score, 
                           recall_score, f1_score, classification_report, confusion_matrix)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# %% 
# Importing Data

df = pd.read_csv('improved_disease_dataset.csv')
df.head()

encoder = LabelEncoder()
df['disease'] = encoder.fit_transform(df['disease'])

X = df.drop('disease', axis = 1)
y = df['disease']

# Add feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

disease_categories = df['disease'].unique()


plt.figure(figsize=(18,8))
sns.countplot(x=y)
plt.title("Disease Class Distribution Before Resampling")
plt.xticks(rotation=90)
plt.show()

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y) # Balance the dataset

print("Resampled Class Distribution:\n", pd.Series(y_resampled).value_counts())

# %%
# Cross - Valdiation with Stratified K-Fold

if 'gender' in X_resampled.columns:
    le = LabelEncoder()
    X_resampled['gender'] = le.fit_transform(X_resampled['gender'])

X_resampled = X_resampled.fillna(0)

if len(y_resampled.shape) > 1:
    y_resampled = y_resampled.values.ravel()

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),  # Added probability=True for predict_proba
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(max_iter=1000, random_state=42),
    'LDA': LinearDiscriminantAnalysis(),
    'Extra Trees': ExtraTreesClassifier(random_state=42)
}

cv_scoring = 'accuracy'
stratified_kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=42)

for name, model in models.items():
    try: 
        scores = cross_val_score(
            model,
            X_resampled,
            y_resampled,
            cv = stratified_kfold,
            scoring = cv_scoring,
            n_jobs =- 1,
            error_score='raise'
        )
        print("=" * 50)
        print(f"Model: {name}")
        print(f"Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f}")
    except Exception as e:
        print("=" * 50)
        print(f"Model: {name} failed with error:")
        print(e)

# %%
# Training Individual Models and Generating Confusion Matrices
svm = SVC()
svm.fit(X_resampled, y_resampled)
svm_preds = svm.predict(X_resampled)

cf_svm = confusion_matrix(y_resampled, svm_preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_svm, annot=True, fmt='d')
plt.title("Confusion Matrix for SVM Classifier")
plt.show()

print(f"SVM Accuracy: {accuracy_score(y_resampled, svm_preds) * 100:.2f}%")

nb = GaussianNB()
nb.fit(X_resampled, y_resampled)
nb_preds = nb.predict(X_resampled)

cf_nb = confusion_matrix(y_resampled, nb_preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_nb, annot=True, fmt='d')
plt.title("Confusion Matrix for Gaussian NB Classifier")
plt.show()

print(f"NB Accuracy: {accuracy_score(y_resampled, nb_preds) * 100:.2f}%")

rf = RandomForestClassifier(random_state=42)
rf.fit(X_resampled, y_resampled)
rf_preds = rf.predict(X_resampled)

cf_rf = confusion_matrix(y_resampled, rf_preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_rf, annot=True, fmt='d')
plt.title("Confusion Matrix for RF Classifier")
plt.show()

print(f"RF Accuracy: {accuracy_score(y_resampled, rf_preds) * 100:.2f}%")


# %%
# Combining Predictions for Robustness

from statistics import mode

final_preds = [mode([i,j,k]) for i,j,k in zip(svm_preds, nb_preds, rf_preds)]

cf_combined = confusion_matrix(y_resampled, final_preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_combined, annot=True, fmt='d')
plt.title("Confusion Matrix for Combined Model")
plt.show()

print(f"Combined Model Accuracy: {accuracy_score(y_resampled, final_preds) * 100:.2f}%")

# %%
# Creating Prediction Function 

symptoms = X.columns.values
symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}

def predict_disease(input_symptoms):
    input_symptoms = input_symptoms.split(",")
    input_data = [0] * len(symptom_index)

    for symptom in input_symptoms:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1
    
    # Create DataFrame with proper feature names
    input_data = pd.DataFrame([input_data], columns=X.columns)
    
    # Scale the input data using the same scaler
    input_data_scaled = scaler.transform(input_data)
    input_data_scaled = pd.DataFrame(input_data_scaled, columns=X.columns)
    
    # Make predictions
    rf_pred = encoder.classes_[rf.predict(input_data_scaled)[0]]
    nb_pred = encoder.classes_[nb.predict(input_data_scaled)[0]]
    svm_pred = encoder.classes_[svm.predict(input_data_scaled)[0]]

    final_pred = mode([rf_pred, nb_pred, svm_pred])

    return {
        "Random Forest Prediction": rf_pred,
        "Naive Bayes Prediction": nb_pred,
        "SVM Prediction": svm_pred,
        "Final Prediction": final_pred
    }

print(predict_disease("Itching, Skin Rash, Nodal Skin Eruptions"))
# %%
# Fine-Tune the models

# Split data before fine-tuning
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to maintain feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'class_weight': [None, 'balanced']
}

param_grid_nb = {
    'var_smoothing': list(np.logspace(0,-9, num=100))
}

param_grid_rf = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
}

print(f"\nFine-tuning SVM...")
svm = SVC()  
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train_scaled, y_train)
print(f"Best Parameters: {grid_search_svm.best_params_}")
print(f"Best CV Score: {grid_search_svm.best_score_:.3f}")
svm = grid_search_svm.best_estimator_

print(f"\nFine-tuning NB...")
nb = GaussianNB()  
grid_search_nb = GridSearchCV(nb, param_grid_nb, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_nb.fit(X_train_scaled, y_train)
print(f"Best Parameters: {grid_search_nb.best_params_}")
print(f"Best CV Score: {grid_search_nb.best_score_:.3f}")
nb = grid_search_nb.best_estimator_

print(f"\nFine-tuning Random Forest...")
rf = RandomForestClassifier(random_state=42)  
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train_scaled, y_train)
print(f"Best Parameters: {grid_search_rf.best_params_}")
print(f"Best CV Score: {grid_search_rf.best_score_:.3f}")
rf = grid_search_rf.best_estimator_

# Evaluate on test set
print("\nEvaluating models on test set:")
svm_test_pred = svm.predict(X_test_scaled)
nb_test_pred = nb.predict(X_test_scaled)
rf_test_pred = rf.predict(X_test_scaled)

print(f"SVM Test Accuracy: {accuracy_score(y_test, svm_test_pred):.3f}")
print(f"NB Test Accuracy: {accuracy_score(y_test, nb_test_pred):.3f}")
print(f"RF Test Accuracy: {accuracy_score(y_test, rf_test_pred):.3f}")

#%%
# Combined predictions on test set
final_test_pred = [mode([i,j,k]) for i,j,k in zip(svm_test_pred, nb_test_pred, rf_test_pred)]
print(f"Combined Model Test Accuracy: {accuracy_score(y_test, final_test_pred):.3f}")

best_model = rf 
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# %%
# Confusion Matrix for final model 
plt.figure(figsize=(12,8))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=disease_categories,
            yticklabels=disease_categories)
plt.title(f'Confusion Matrix - {best_model}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# %%
# Feature Selection and Engineering
print("\nModel Improvement Analysis:")
print("-" * 50)

# Feature Selection
print("\n Feature Selection Analysis:")
# Select top 95% of features based on variance
selector = SelectKBest(f_classif, k=int(len(X_train_scaled.columns) * 0.95))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_features = X_train_scaled.columns[selector.get_support()].tolist()
print(f"\nSelected {len(selected_features)} features:")
print(selected_features)

feature_scores = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Score': selector.scores_
})
feature_scores = feature_scores.sort_values('Score', ascending=False)
print("\nFeature Importance Scores:")
print(feature_scores)

plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Feature', data=feature_scores)
plt.title('Feature Importance Scores')
plt.tight_layout()
plt.show()

# Update best model with selected features
best_model.fit(X_train_selected, y_train)
y_pred = best_model.predict(X_test_selected)
y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]

print(f"\nModel accuracy with selected features: {accuracy_score(y_test, y_pred):.3f}")

# %%
# PCA Analysis
print("\nPCA ANAlysis:")
pca = PCA()
X_pca = pca.fit_transform(X_train_scaled)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10,6))
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, 'bo-')
plt.axhline(y = 0.95, color = 'r', linestyle ='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.show()
# %%
# Different Sampling Strategies

print("\n Testing Different Sampling Strategies:")

samplers = {
    'Original' : None,
    'SMOTE' : SMOTE(random_state=42),
    'RandomUnderSampler': RandomUnderSampler(random_state=42)
}

for name, sampler in samplers.items():
    if sampler is None: 
        X_resampled, y_resampled = X_train, y_train
    else:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    model = best_model.__class__(**best_model.get_params())
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{name:20} Accuracy: {score:.3f}")


# %%
# Recommendations based on analysis

print("\nRecommendations for Model Improvement:")
print("-" * 50)

# Feature selection recommendation  
print("\n Feature Selection:")
print(f"Consider using only the top {sum(cumulative_variance < 0.95)} features that explain 95% of variance")
print("Top 5 most important features:")
print(feature_scores.head().to_string())

# %%
# Overfitting Analysis and Validation

print("\n Overfitting Analysis:")

# Learning Curves
print("\n Learning Curves:")
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train_scaled, y_train, cv=5, scoring='accuracy', 
    n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
plt.plot(train_sizes, test_mean, 'o-', color = 'g', label = 'Cross-Validation Score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha = 0.1, color = 'r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha = 0.1, color = 'g')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.show()

# %%
# Validation Curves

print("\n Validation Curves:")
param_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
param_name = 'n_estimators'
train_scores, test_scores = validation_curve(
    best_model, X_train_scaled, y_train, param_name = param_name, 
    param_range = param_range, cv = 5, scoring = 'accuracy', 
    n_jobs = -1
)

train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

plt.figure(figsize=(10,6))
plt.plot(param_range, train_mean, 'o-', color = 'r', label = 'Training Score')
plt.plot(param_range, test_mean, 'o-', color = 'g', label = 'Cross-Validation Score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha = 0.1, color = 'r')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha = 0.1, color = 'g')
plt.xlabel(param_name)
plt.ylabel('Accuracy')
plt.title('Validation Curve')
plt.legend()
plt.show()

# %%
# Confidence vs Accuracy

print("\n Confidence vs Accuracy:")
confidence_bins = np.linspace(0, 1.0, 11)
accuracy_by_confidence = []

for i in range(len(confidence_bins) - 1):
    mask = (y_pred_proba >= confidence_bins[i]) & (y_pred_proba < confidence_bins[i+1])
    if np.sum(mask) > 0:
        accuracy = accuracy_score(y_test[mask], y_pred[mask])
        accuracy_by_confidence.append(accuracy)
    else:
        accuracy_by_confidence.append(0)

plt.figure(figsize=(10,6))
plt.plot(confidence_bins[:-1], accuracy_by_confidence, 'o-', color = 'b')
plt.plot([0,1], [0, 1], 'k--', label = 'Perfect Calibration')
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title('Confidence vs Accuracy')
plt.legend()
plt.show()

#%% 
# Bootstrap Analysis
n_iterations = 100
bootstrap_scores = []

y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
X_train_array = X_train_scaled.values if hasattr(X_train_scaled, 'values') else X_train_scaled
X_test_array = X_test_scaled.values if hasattr(X_test_scaled, 'values') else X_test_scaled

for i in range(n_iterations):
    indices = np.random.randint(0, len(X_train_array), len(X_train_array))
    X_bootstrap = X_train_array[indices]
    y_bootstrap = y_train_array[indices]

    best_model.fit(X_bootstrap, y_bootstrap)
    score = best_model.score(X_test_array, y_test_array)
    bootstrap_scores.append(score)

confidence_interval = np.percentile(bootstrap_scores, [2.5, 97.5])
print(f"Bootstrap 95% Confidence Interval: {confidence_interval[0]:.3f} to {confidence_interval[1]:.3f}")

# %%

    





# %%
