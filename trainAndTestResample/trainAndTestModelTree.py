# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
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

# df = pd.read_csv("./data/ufeaturesPDRed13EV41kHzStride2.csv") df = pd.read_csv("./data/selected_ThresholdfeaturesPDRed13EV44,1kHzStride2.csv")
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
    'trainingModel__criterion': ['gini', 'entropy'],
    'trainingModel__splitter': ['best', 'random'],
    'trainingModel__max_depth': [None, 5, 10, 15, 20],
    'trainingModel__min_samples_split': [2, 5, 10],
    'trainingModel__min_samples_leaf': [1, 2, 4],
    'trainingModel__max_features': ['sqrt', 'log2', None],
    'trainingModel__max_leaf_nodes': [None, 10, 20, 30],
    'trainingModel__min_impurity_decrease': [0.0, 0.1, 0.2],
    'trainingModel__class_weight': [None],
}

model = DecisionTreeClassifier(random_state=42)
sm = SVMSMOTE()
trainModelWithSampling(X_train, y_train, X_test, y_test, preprocessor, sm, model, param_grid, CV, N_JOBS)
# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2
# Best Parameters: {'samplingModel__k_neighbors': 4, 'samplingModel__m_neighbors': 6, 'samplingModel__sampling_strategy': {0: 40, 1: 40}, 'trainingModel__class_weight': None, 'trainingModel__criterion': 'gini', 'trainingModel__max_depth': 5, 'trainingModel__max_features': 'sqrt', 'trainingModel__max_leaf_nodes': 20, 'trainingModel__min_impurity_decrease': 0.0, 'trainingModel__min_samples_leaf': 1, 'trainingModel__min_samples_split': 2, 'trainingModel__splitter': 'best'}
# Best Score: 0.8
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.67      0.60      0.63        10
#            1       0.64      0.70      0.67        10

#     accuracy                           0.65        20
#    macro avg       0.65      0.65      0.65        20
# weighted avg       0.65      0.65      0.65        20


# Confusion Matrix:
#  [[6 4]
#  [3 7]]