from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier 
import matplotlib
from imblearn.over_sampling import SVMSMOTE
from utils import trainModel
CV = 5
N_JOBS = -1
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
# # Resample the training dataset to improve predictions
# sm = SVMSMOTE(sampling_strategy={0: 100, 1: 100}, m_neighbors=5, k_neighbors=4)
# X_train, y_train = sm.fit_resample(X_train,y_train)
 # Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10,25,50,75,100,150],
    #   * **n_estimators:** The number of weak learners (base estimators) to train.
    #   * Increasing `n_estimators` can improve performance, but also increases computational cost.
    #   * Too many estimators can lead to overfitting, so it's important to tune this parameter.

    'learning_rate': [0.01, 0.05, 0.1, 0.2, 1.0],
    #   * **learning_rate:**
    #       * Controls the contribution of each weak learner to the final prediction.
    #       * Smaller values require more `n_estimators` but can improve generalization.
    #       * Typical range: 0.01 to 0.2.
    #       * A value of 1.0 means no shrinkage.
    'estimator': [DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=5), DecisionTreeClassifier(max_depth=7), DecisionTreeClassifier(max_depth=9)],

}

model = AdaBoostClassifier(random_state=42)

trainModel(X_train, y_train,X_test, y_test, preprocessor, model, param_grid, CV, N_JOBS)

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
# RESAMPLE 0.2 
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

# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2
# Best Parameters: {'estimator': DecisionTreeClassifier(max_depth=1), 'learning_rate': 0.1, 'n_estimators': 100}
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

# Only U
# NO RESAMPLE
# Best Parameters: {'estimator': DecisionTreeClassifier(max_depth=1), 'learning_rate': 0.05, 'n_estimators': 75}
# Best Score: 0.725
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.75      0.30      0.43        10
#            1       0.56      0.90      0.69        10

#     accuracy                           0.60        20
#    macro avg       0.66      0.60      0.56        20
# weighted avg       0.66      0.60      0.56        20


# Confusion Matrix:
#  [[3 7]
#  [1 9]]