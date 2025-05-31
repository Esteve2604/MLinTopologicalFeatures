from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib
from imblearn.over_sampling import SVMSMOTE

CV = 5
# df = pd.read_csv("./data/featuresPDRed2.5EV8kHzStride2.csv")
# df = pd.read_csv("./data/featuresPDRed13EV41kHzStride2.csv")
# df = pd.read_csv("./data/ufeaturesPDRed2.5EV8kHzStride2.csv")
df = pd.read_csv("./data/ufeaturesPDRed13EV41kHzStride2.csv")
df.drop("sampleName", axis=1, inplace=True)

y = df["parkinson?"]
x = df.drop("parkinson?", axis=1)

# Create a ColumnTransformer to apply different preprocessing to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), x.columns.to_numpy()),
        # ('scale', MaxAbsScaler(), x.columns.to_numpy())
    ])




X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y
    )

# Resample the training dataset to improve predictions
# sm = SVMSMOTE(sampling_strategy={0: 100, 1: 100}, m_neighbors=5, k_neighbors=4)
# X_train, y_train = sm.fit_resample(X_train,y_train)

 # Define the parameter grid for GridSearchCV
param_grid = {
    'loss': ['log_loss'],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_iter': [50, 100, 200, 300],
    'max_depth': [None, 3, 5, 7],
    'min_samples_leaf': [5, 10, 20],
    'l2_regularization': [0.0, 0.1, 0.001],
    'early_stopping': [True],
    'n_iter_no_change': [10],
    'validation_fraction': [0.1],
    'class_weight': [None]
}

tree_classifier = HistGradientBoostingClassifier(random_state=42)
# Use GridSearchCV with the custom scorer
grid_search = GridSearchCV(
    tree_classifier,
    param_grid,
    scoring='accuracy', 
    cv=CV,
    n_jobs=-1,
    verbose=1
)
pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('gridSearch', grid_search)])
pipeline.fit(X_train, y_train)

#Extracting best score and parameters
best_params = pipeline.named_steps['gridSearch'].best_params_
best_score = pipeline.named_steps['gridSearch'].best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)
best_tree = pipeline.named_steps['gridSearch'].best_estimator_

#Training the best model obtained
X_test = preprocessor.transform(X_test)
y_pred = best_tree.predict(X_test)

# Calculate evaluation metrics
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
# Print the metrics
print("Validation Metrics:")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

# NO RESAMPLE 0.2 NOT STANDARIZED
# Best Parameters: {'class_weight': None, 'early_stopping': True, 'l2_regularization': 0.1, 'learning_rate': 0.01, 'loss': 'log_loss', 'max_depth': None, 'max_iter': 100, 'min_samples_leaf': 5, 'n_iter_no_change': 10, 'validation_fraction': 0.1}
# Best Score: 0.7160714285714286
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.56      0.56      0.56         9
#            1       0.60      0.60      0.60        10

#     accuracy                           0.58        19
#    macro avg       0.58      0.58      0.58        19
# weighted avg       0.58      0.58      0.58        19


# Confusion Matrix:
#  [[5 4]
#  [4 6]]
# NO RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'class_weight': None, 'early_stopping': True, 'l2_regularization': 0.0, 'learning_rate': 0.01, 'loss': 'log_loss', 'max_depth': None, 'max_iter': 50, 'min_samples_leaf': 20, 'n_iter_no_change': 10, 'validation_fraction': 0.1}
# Best Score: 0.6839285714285713
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.75      0.67      0.71         9
#            1       0.73      0.80      0.76        10

#     accuracy                           0.74        19
#    macro avg       0.74      0.73      0.73        19
# weighted avg       0.74      0.74      0.74        19


# Confusion Matrix:
#  [[6 3]
#  [2 8]]

# NO RESAMPLE 0.2 SCALED
# Best Parameters: {'class_weight': None, 'early_stopping': True, 'l2_regularization': 0.1, 'learning_rate': 0.05, 'loss': 'log_loss', 'max_depth': 3, 'max_iter': 50, 'min_samples_leaf': 5, 'n_iter_no_change': 10, 'validation_fraction': 0.1}
# Best Score: 0.6892857142857143
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.40      0.22      0.29         9
#            1       0.50      0.70      0.58        10

#     accuracy                           0.47        19
#    macro avg       0.45      0.46      0.43        19
# weighted avg       0.45      0.47      0.44        19


# Confusion Matrix:
#  [[2 7]
#  [3 7]]
# NO RESAMPLE STANDARIZED AND SCALED
# Best Parameters: {'class_weight': None, 'early_stopping': True, 'l2_regularization': 0.1, 'learning_rate': 0.2, 'loss': 'log_loss', 'max_depth': None, 'max_iter': 50, 'min_samples_leaf': 10, 'n_iter_no_change': 10, 'validation_fraction': 0.1}
# Best Score: 0.6928571428571428
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.60      0.67      0.63         9
#            1       0.67      0.60      0.63        10

#     accuracy                           0.63        19
#    macro avg       0.63      0.63      0.63        19
# weighted avg       0.64      0.63      0.63        19


# Confusion Matrix:
#  [[6 3]
#  [4 6]]
# RESAMPLE 100 samples 0.2
# Best Parameters: {'class_weight': None, 'early_stopping': True, 'l2_regularization': 0.0, 'learning_rate': 0.2, 'loss': 'log_loss', 'max_depth': 5, 'max_iter': 50, 'min_samples_leaf': 5, 'n_iter_no_change': 10, 'validation_fraction': 0.1}
# Best Score: 0.9049999999999999
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.67      0.67      0.67         9
#            1       0.70      0.70      0.70        10

#     accuracy                           0.68        19
#    macro avg       0.68      0.68      0.68        19
# weighted avg       0.68      0.68      0.68        19


# Confusion Matrix:
#  [[6 3]
#  [3 7]]

# pDRed2.5EV8kHzStride2
# NO RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'class_weight': None, 'early_stopping': True, 'l2_regularization': 0.1, 'learning_rate': 0.05, 'loss': 'log_loss', 'max_depth': 3, 'max_iter': 200, 'min_samples_leaf': 10, 'n_iter_no_change': 10, 'validation_fraction': 0.1}
# Best Score: 0.7625
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.58      0.70      0.64        10
#            1       0.62      0.50      0.56        10

#     accuracy                           0.60        20
#    macro avg       0.60      0.60      0.60        20
# weighted avg       0.60      0.60      0.60        20


# Confusion Matrix:
#  [[7 3]
#  [5 5]]
# RESAMPLE 0.2 STANDARIZED

# Best Score: 0.9199999999999999
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.78      0.70      0.74        10
#            1       0.73      0.80      0.76        10

#     accuracy                           0.75        20
#    macro avg       0.75      0.75      0.75        20
# weighted avg       0.75      0.75      0.75        20


# Confusion Matrix:
#  [[7 3]
#  [2 8]]

# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2
# Best Parameters: {'class_weight': None, 'early_stopping': True, 'l2_regularization': 0.0, 'learning_rate': 0.2, 'loss': 'log_loss', 'max_depth': None, 'max_iter': 50, 'min_samples_leaf': 10, 'n_iter_no_change': 10, 'validation_fraction': 0.1}
# Best Score: 0.7
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.75      0.60      0.67        10
#            1       0.67      0.80      0.73        10

#     accuracy                           0.70        20
#    macro avg       0.71      0.70      0.70        20
# weighted avg       0.71      0.70      0.70        20


# Confusion Matrix:
#  [[6 4]
#  [2 8]]

# ONLY U
# No RESAMPLE
# Best Parameters: {'class_weight': None, 'early_stopping': True, 'l2_regularization': 0.0, 'learning_rate': 0.2, 'loss': 'log_loss', 'max_depth': None, 'max_iter': 50, 'min_samples_leaf': 20, 'n_iter_no_change': 10, 'validation_fraction': 0.1}
# Best Score: 0.725
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.75      0.60      0.67        10
#            1       0.67      0.80      0.73        10

#     accuracy                           0.70        20
#    macro avg       0.71      0.70      0.70        20
# weighted avg       0.71      0.70      0.70        20


# Confusion Matrix:
#  [[6 4]
#  [2 8]]