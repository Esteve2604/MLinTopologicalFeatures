# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib
from imblearn.over_sampling import SVMSMOTE
from utils import trainModelWithSampling

CV = 5
N_JOBS = 2
# df = pd.read_csv("./data/featuresPDRed2.5EV8kHzStride2.csv")
# df = pd.read_csv("./data/featuresPDRed13EV41kHzStride2.csv")

# df = pd.read_csv("./data/ufeaturesPDRed13EV41kHzStride2.csv") 
df = pd.read_csv("./data/selected_ThresholdfeaturesPDRed13EV44,1kHzStride2.csv")
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
# sm = SVMSMOTE(sampling_strategy={0: 100, 1: 100}, m_neighbors=5, k_neighbors=4)
# X_train, y_train = sm.fit_resample(X_train,y_train)

param_grid = {
    'samplingModel__sampling_strategy': [{0: 40, 1: 40}],
    'samplingModel__m_neighbors': [5],
    'samplingModel__k_neighbors': [5],
    'trainingModel__C': [0.001, 0.01, 0.1], #[0.001, 0.01, 0.1, 1, 10, 100]
    'trainingModel__kernel': ['linear', 'rbf','poly','sigmoid'],
    'trainingModel__degree': [2,3],
    'trainingModel__gamma': ['scale', 'auto', 0.1, 1],
    'trainingModel__coef0': [0.0, 0.1, 0.5, 1.0],
    'trainingModel__class_weight': [None],
    'trainingModel__probability': [False]
}

model = SVC(random_state=42)
sm = SVMSMOTE()
trainModelWithSampling(X_train, y_train, X_test, y_test, preprocessor, sm, model, param_grid, CV, N_JOBS)


# featuresPDRed13EV41kHzStride2
# RESAMPLE 0.2

# Best Parameters: {'samplingModel__k_neighbors': 5, 'samplingModel__m_neighbors': 5, 'samplingModel__sampling_strategy': {0: 40, 1: 40}, 'trainingModel__C': 10, 'trainingModel__class_weight': None, 'trainingModel__coef0': 0.1, 'trainingModel__degree': 3, 'trainingModel__gamma': 'auto', 'trainingModel__kernel': 'sigmoid', 'trainingModel__probability': False}
# Best Score: 0.7625
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.57      0.80      0.67        10
#            1       0.67      0.40      0.50        10

#     accuracy                           0.60        20
#    macro avg       0.62      0.60      0.58        20
# weighted avg       0.62      0.60      0.58        20


# Confusion Matrix:
#  [[8 2]
#  [6 4]]
# Best Parameters: {'samplingModel__k_neighbors': 5, 'samplingModel__m_neighbors': 5, 'samplingModel__sampling_strategy': {0: 40, 1: 40}, 'trainingModel__C': 0.01, 'trainingModel__class_weight': None, 'trainingModel__coef0': 0.1, 'trainingModel__degree': 2, 'trainingModel__gamma': 'auto', 'trainingModel__kernel': 'linear', 'trainingModel__probability': False}
# Best Score: 0.7
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.75      0.90      0.82        10
#            1       0.88      0.70      0.78        10

#     accuracy                           0.80        20
#    macro avg       0.81      0.80      0.80        20
# weighted avg       0.81      0.80      0.80        20


# Confusion Matrix:
#  [[9 1]
#  [3 7]]