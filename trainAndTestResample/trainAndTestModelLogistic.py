# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.linear_model import LogisticRegression
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
 # Define the parameter grid for GridSearchCV
param_grid = {
    'samplingModel__sampling_strategy': [{0: 40, 1: 40}],
    'samplingModel__m_neighbors': [4,5,6],
    'samplingModel__k_neighbors': [3,4,5],
    'trainingModel__penalty': ['elasticnet'],
    'trainingModel__C': [0.001, 0.01, 0.1, 1, 10],
    'trainingModel__solver': ['saga'],
    'trainingModel__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
    'trainingModel__class_weight': [None],
    'trainingModel__max_iter': [750, 1250, 2500, 5000, 7500, 10000],
}

model = LogisticRegression(random_state=42)
sm = SVMSMOTE()
trainModelWithSampling(X_train, y_train, X_test, y_test, preprocessor, sm, model, param_grid, CV, N_JOBS)

# featuresPDRed13EV41kHzStride2
#  RESAMPLE 0.2
# Best Parameters: {'samplingModel__k_neighbors': 3, 'samplingModel__m_neighbors': 4, 'samplingModel__sampling_strategy': {0: 40, 1: 40}, 'trainingModel__C': 0.1, 'trainingModel__class_weight': None, 'trainingModel__l1_ratio': 0.25, 'trainingModel__max_iter': 5000, 'trainingModel__penalty': 'elasticnet', 'trainingModel__solver': 'saga'}
# Best Score: 0.7375
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.67      0.80      0.73        10
#            1       0.75      0.60      0.67        10

#     accuracy                           0.70        20
#    macro avg       0.71      0.70      0.70        20
# weighted avg       0.71      0.70      0.70        20


# Confusion Matrix:
#  [[8 2]
#  [4 6]]