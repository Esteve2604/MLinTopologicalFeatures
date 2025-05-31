from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib
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
        ('num', StandardScaler(), x.columns.to_numpy()),
    ])



X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y
    )
# Resample the training dataset to improve predictions
sm = SVMSMOTE()
 # Define the parameter grid for GridSearchCV
param_grid = {
    'samplingModel__sampling_strategy': [{0: 40, 1: 40}],
    'samplingModel__m_neighbors': [4,5,6],
    'samplingModel__k_neighbors': [3,4,5],
    'trainingModel__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'trainingModel__weights': ['uniform', 'distance'],
    'trainingModel__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'trainingModel__p': [1, 2, 3],
    'trainingModel__leaf_size': [10, 20, 30, 40, 50]
}

model = KNeighborsClassifier()

trainModelWithSampling(X_train, y_train, X_test, y_test, preprocessor, sm, model, param_grid, CV, N_JOBS)



# featuresPDRed13EV41kHzStride2
# RESAMPLE 0.2 Resultados muy inestables en la validación
# Best Parameters: {'samplingModel__k_neighbors': 3, 'samplingModel__m_neighbors': 5, 'samplingModel__sampling_strategy': {0: 40, 1: 40}, 'trainingModel__algorithm': 'kd_tree', 'trainingModel__leaf_size': 30, 'trainingModel__n_neighbors': 11, 'trainingModel__p': 1, 'trainingModel__weights': 'uniform'}
# Best Score: 0.725
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.80      0.80      0.80        10
#            1       0.80      0.80      0.80        10

#     accuracy                           0.80        20
#    macro avg       0.80      0.80      0.80        20
# weighted avg       0.80      0.80      0.80        20


# Confusion Matrix:
#  [[8 2]
#  [2 8]]
# Best Parameters: {'samplingModel__k_neighbors': 3, 'samplingModel__m_neighbors': 4, 'samplingModel__sampling_strategy': {0: 40, 1: 40}, 'trainingModel__algorithm': 'auto', 'trainingModel__leaf_size': 50, 'trainingModel__n_neighbors': 3, 'trainingModel__p': 3, 'trainingModel__weights': 'uniform'}
# Best Score: 0.7625
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.60      0.60      0.60        10
#            1       0.60      0.60      0.60        10

#     accuracy                           0.60        20
#    macro avg       0.60      0.60      0.60        20
# weighted avg       0.60      0.60      0.60        20


# Confusion Matrix:
#  [[6 4]
#  [4 6]]