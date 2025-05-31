from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier 
from imblearn.over_sampling import SVMSMOTE
from utils import trainModelWithSampling

CV = 5
N_JOBS = 2
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
        ('num', StandardScaler(), x.columns.to_numpy())
        # ('scale', MaxAbsScaler(), x.columns.to_numpy())
    ])


X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y
    )
# Resample the training dataset to improve predictions
sm = SVMSMOTE()
 # Define the parameter grid for GridSearchCV
param_grid = { 
              
    'samplingModel__sampling_strategy': [{0: 40, 1: 40}], # {{0: 40, 1: 40}, {0: 50, 1: 50}, {0: 60, 1: 60}, {0: 70, 1: 70}, {0: 80, 1: 80}}
    'samplingModel__m_neighbors': [4,5,6],
    'samplingModel__k_neighbors': [3,4,5],
            
    'trainingModel__n_estimators': [10,25,50,75,100,150, 200, 250],
    #   * **n_estimators:** The number of weak learners (base estimators) to train.
    #   * Increasing `n_estimators` can improve performance, but also increases computational cost.
    #   * Too many estimators can lead to overfitting, so it's important to tune this parameter.

    'trainingModel__learning_rate': [0.01, 0.05, 0.1],
    #   * **learning_rate:**
    #       * Controls the contribution of each weak learner to the final prediction.
    #       * Smaller values require more `n_estimators` but can improve generalization.
    #       * Typical range: 0.01 to 0.2.
    #       * A value of 1.0 means no shrinkage.
    'trainingModel__estimator': [DecisionTreeClassifier()],
    'trainingModel__estimator__max_depth': [1,3,5,7,9],
    'trainingModel__estimator__criterion': ['gini', 'entropy'],
    'trainingModel__estimator__splitter': ['best', 'random'],
    'trainingModel__estimator__max_depth': [None, 10, 20],
    'trainingModel__estimator__min_samples_split': [4],
    'trainingModel__estimator__min_samples_leaf': [2],
    'trainingModel__estimator__max_features': ['sqrt', 'log2', None],
    'trainingModel__estimator__max_leaf_nodes': [None],
    'trainingModel__estimator__min_impurity_decrease': [0.1, 0.2],
    'trainingModel__estimator__class_weight': [None],
}

model = AdaBoostClassifier(random_state=42)
trainModelWithSampling(X_train, y_train, X_test, y_test, preprocessor, sm, model, param_grid, CV, N_JOBS)

# pD4000distRed5EV8kHzStride2
# NO RESAMPLE, 0.2 
# Best Parameters: {'estimator': DecisionTreeClassifier(max_depth=5), 'learning_rate': 1.0, 'n_estimators': 25}
# Best Score: 0.6392857142857143
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
# NO RESAMPLE, 0.2 test Standarized
# Best Parameters: {'estimator': DecisionTreeClassifier(max_depth=1), 'learning_rate': 0.05, 'n_estimators': 75}
# Best Score: 0.759047619047619
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.80      0.44      0.57         9
#            1       0.64      0.90      0.75        10

#     accuracy                           0.68        19
#    macro avg       0.72      0.67      0.66        19
# weighted avg       0.72      0.68      0.67        19

# NO RESAMPLE, 0.2 test Scaled
# Best Parameters: {'estimator': DecisionTreeClassifier(max_depth=1), 'learning_rate': 0.01, 'n_estimators': 50}
# Best Score: 0.7107142857142857
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.80      0.44      0.57         9
#            1       0.64      0.90      0.75        10

#     accuracy                           0.68        19
#    macro avg       0.72      0.67      0.66        19
# weighted avg       0.72      0.68      0.67        19


# Confusion Matrix:
#  [[4 5]
#  [1 9]]

# NO RESAMPLE, 0.2 test Standarized and scaled
# Best Parameters: {'estimator': DecisionTreeClassifier(max_depth=1), 'learning_rate': 0.05, 'n_estimators': 75}
# Best Score: 0.759047619047619
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.80      0.44      0.57         9
#            1       0.64      0.90      0.75        10

#     accuracy                           0.68        19
#    macro avg       0.72      0.67      0.66        19
# weighted avg       0.72      0.68      0.67        19

# Confusion Matrix:
#  [[4 5]
#  [1 9]]

# NO RESAMPLE 0.5 test

# Best Parameters: {'estimator': DecisionTreeClassifier(max_depth=1), 'learning_rate': 0.1, 'n_estimators': 100}
# Best Score: 0.7377777777777778
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.76      0.57      0.65        23
#            1       0.67      0.83      0.74        24

#     accuracy                           0.70        47
#    macro avg       0.72      0.70      0.70        47
# weighted avg       0.71      0.70      0.70        47


# Confusion Matrix:
#  [[13 10]
#  [ 4 20]]



# Confusion Matrix:
#  [[19  4]
#  [10 14]]

# RESAMPLE 0.2
# Best Parameters: {'estimator': DecisionTreeClassifier(max_depth=3), 'learning_rate': 1.0, 'n_estimators': 150}
# Best Score: 0.9400000000000001
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.86      0.67      0.75         9
#            1       0.75      0.90      0.82        10

#     accuracy                           0.79        19
#    macro avg       0.80      0.78      0.78        19
# weighted avg       0.80      0.79      0.79        19


# Confusion Matrix:
#  [[6 3]
#  [1 9]]

# pDRed2.5EV8kHzStride2
# NO RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'estimator': DecisionTreeClassifier(max_depth=3), 'learning_rate': 0.1, 'n_estimators': 75}
# Best Score: 0.7375
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.73      0.80      0.76        10
#            1       0.78      0.70      0.74        10

#     accuracy                           0.75        20
#    macro avg       0.75      0.75      0.75        20
# weighted avg       0.75      0.75      0.75        20


# Confusion Matrix:
#  [[8 2]
#  [3 7]]
# RESAMPLE 0.2 {0:100, 1:100}
# Best Parameters: {'estimator': DecisionTreeClassifier(max_depth=3), 'learning_rate': 1.0, 'n_estimators': 100}
# Best Score: 0.95
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.80      0.80      0.80        10
#            1       0.80      0.80      0.80        10

#     accuracy                           0.80        20
#    macro avg       0.80      0.80      0.80        20
# weighted avg       0.80      0.80      0.80        20
# RESAMPLE 0.2 {0:60, 1:60}
# Best Parameters: {'samplingModel__k_neighbors': 4, 'samplingModel__m_neighbors': 4, 'samplingModel__sampling_strategy': {0: 60, 1: 60}, 'trainingModel__estimator': DecisionTreeClassifier(max_depth=3), 'trainingModel__learning_rate': 0.1, 'trainingModel__n_estimators': 50}
# Best Score: 0.7625
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.64      0.70      0.67        10
#            1       0.67      0.60      0.63        10

#     accuracy                           0.65        20
#    macro avg       0.65      0.65      0.65        20
# weighted avg       0.65      0.65      0.65        20


# Confusion Matrix:
#  [[7 3]
#  [4 6]]

# RESAMPLE 0.2 {0:40, 1:40}
# Best Parameters: {'samplingModel__k_neighbors': 3, 'samplingModel__m_neighbors': 4, 'samplingModel__sampling_strategy': {0: 40, 1: 40}, 'trainingModel__estimator': DecisionTreeClassifier(max_depth=7), 'trainingModel__learning_rate': 0.1, 'trainingModel__n_estimators': 150}
# Best Score: 0.7625
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.73      0.80      0.76        10
#            1       0.78      0.70      0.74        10

#     accuracy                           0.75        20
#    macro avg       0.75      0.75      0.75        20
# weighted avg       0.75      0.75      0.75        20


# Confusion Matrix:
#  [[8 2]
#  [3 7]]

# Best Parameters: {'samplingModel__k_neighbors': 3, 'samplingModel__m_neighbors': 4, 'samplingModel__sampling_strategy': {0: 40, 1: 40}, 'trainingModel__estimator': DecisionTreeClassifier(max_depth=9), 'trainingModel__learning_rate': 1.0, 'trainingModel__n_estimators': 50}
# Best Score: 0.8375
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.44      0.40      0.42        10
#            1       0.45      0.50      0.48        10

#     accuracy                           0.45        20
#    macro avg       0.45      0.45      0.45        20
# weighted avg       0.45      0.45      0.45        20


# Confusion Matrix:
#  [[4 6]
#  [5 5]]

# Best Parameters: {'samplingModel__k_neighbors': 4, 'samplingModel__m_neighbors': 5, 'samplingModel__sampling_strategy': {0: 40, 1: 40}, 'trainingModel__estimator': DecisionTreeClassifier(max_depth=5), 'trainingModel__learning_rate': 0.01, 'trainingModel__n_estimators': 50}
# Best Score: 0.8
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.67      0.40      0.50        10
#            1       0.57      0.80      0.67        10

#     accuracy                           0.60        20
#    macro avg       0.62      0.60      0.58        20
# weighted avg       0.62      0.60      0.58        20


# Confusion Matrix:
#  [[4 6]
#  [2 8]]

# Best Parameters: {'samplingModel__k_neighbors': 4, 'samplingModel__m_neighbors': 5, 'samplingModel__sampling_strategy': {0: 40, 1: 40}, 'trainingModel__estimator': DecisionTreeClassifier(max_depth=3), 'trainingModel__learning_rate': 0.05, 'trainingModel__n_estimators': 25}
# Best Score: 0.7625
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.55      0.60      0.57        10
#            1       0.56      0.50      0.53        10

#     accuracy                           0.55        20
#    macro avg       0.55      0.55      0.55        20
# weighted avg       0.55      0.55      0.55        20


# Confusion Matrix:
#  [[6 4]
#  [5 5]]

# featuresPDRed13EV41kHzStride2
# RESAMPLE 0.2
# Best Parameters: {'samplingModel__k_neighbors': 3, 'samplingModel__m_neighbors': 5, 'samplingModel__sampling_strategy': {0: 40, 1: 40}, 'trainingModel__estimator': DecisionTreeClassifier(), 'trainingModel__estimator__class_weight': None, 'trainingModel__estimator__criterion': 'entropy', 'trainingModel__estimator__max_depth': 20, 'trainingModel__estimator__max_features': None, 'trainingModel__estimator__max_leaf_nodes': None, 'trainingModel__estimator__min_impurity_decrease': 0.1, 'trainingModel__estimator__min_samples_leaf': 2, 'trainingModel__estimator__min_samples_split': 4, 'trainingModel__estimator__splitter': 'best', 'trainingModel__learning_rate': 0.1, 'trainingModel__n_estimators': 100}
# Best Score: 0.75
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.70      0.70      0.70        10
#            1       0.70      0.70      0.70        10

#     accuracy                           0.70        20
#    macro avg       0.70      0.70      0.70        20
# weighted avg       0.70      0.70      0.70        20


# Confusion Matrix:
#  [[7 3]
#  [3 7]]

# ONLY U
# NO RESAMPLE 0.2