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
from utils import selectFeaturesSFSAndTrainModel

N_FEATURES_TO_SELECT = 15
CV = 2
N_JOBS = 2

# df = pd.read_csv("./data/featuresPDRed2.5EV8kHzStride2.csv")
# df = pd.read_csv("./data/featuresPDRed13EV41kHzStride2.csv")

# datasetPath = "./data/afeaturesPDRed13EV41kHzStride2.csv"
# datasetPath = "./data/efeaturesPDRed13EV41kHzStride2.csv"
# datasetPath = "./data/ifeaturesPDRed13EV41kHzStride2.csv"
# datasetPath = "./data/ofeaturesPDRed13EV41kHzStride2.csv"
datasetPath = "./data/ufeaturesPDRed13EV41kHzStride2.csv"

df = pd.read_csv(datasetPath)
print("El dataset utilizado es: " + datasetPath)

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
# sm = SVMSMOTE(sampling_strategy={0: 40, 1: 40}, m_neighbors=5, k_neighbors=4)
# X_train, y_train = sm.fit_resample(X_train,y_train)

model = HistGradientBoostingClassifier(random_state=42, class_weight=None, early_stopping=True, l2_regularization=0.0, learning_rate=0.01, loss='log_loss', max_depth=None, max_iter=50, min_samples_leaf=20, n_iter_no_change=10, validation_fraction=0.1)

selectFeaturesSFSAndTrainModel(X_train, y_train,X_test, y_test, preprocessor, model, N_FEATURES_TO_SELECT, CV, N_JOBS)

# ___SFS____:

# Nº of features selected: 15
# Selected features: ['UcomplexPolinomialCoef6H1', 'UcomplexPolinomialCoef7H1', 'UcomplexPolinomialCoef8H1', 'UcomplexPolinomialCoef9H1', 'UcomplexPolinomialCoef10H1', 'UcomplexPolinomialCoef1H2', 'UcomplexPolinomialCoef2H2', 'UcomplexPolinomialCoef3H2', 'UcomplexPolinomialCoef4H2', 'UcomplexPolinomialCoef5H2', 'UcomplexPolinomialCoef6H2', 'UcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.86      0.60      0.71        10
#            1       0.69      0.90      0.78        10

#     accuracy                           0.75        20
#    macro avg       0.77      0.75      0.74        20
# weighted avg       0.77      0.75      0.74        20

# Confusion Matrix:
#  [[6 4]
#  [1 9]]

# ___SFS____:

# Nº of features selected: 15
# Selected features: ['UcomplexPolinomialCoef6H1', 'UcomplexPolinomialCoef7H1', 'UcomplexPolinomialCoef8H1', 'UcomplexPolinomialCoef9H1', 'UcomplexPolinomialCoef10H1', 'UcomplexPolinomialCoef1H2', 'UcomplexPolinomialCoef2H2', 'UcomplexPolinomialCoef3H2', 'UcomplexPolinomialCoef4H2', 'UcomplexPolinomialCoef5H2', 'UcomplexPolinomialCoef6H2', 'UcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       1.00      0.50      0.67        10
#            1       0.67      1.00      0.80        10

#     accuracy                           0.75        20
#    macro avg       0.83      0.75      0.73        20
# weighted avg       0.83      0.75      0.73        20


# Confusion Matrix:
#  [[ 5  5]
#  [ 0 10]]
# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2

# ___SFS____:

# Nº of features selected: 15
# Selected features: ['UcomplexPolinomialCoef6H1', 'UcomplexPolinomialCoef7H1', 'UcomplexPolinomialCoef8H1', 'UcomplexPolinomialCoef9H1', 'UcomplexPolinomialCoef10H1', 'UcomplexPolinomialCoef1H2', 'UcomplexPolinomialCoef2H2', 'UcomplexPolinomialCoef3H2', 'UcomplexPolinomialCoef4H2', 'UcomplexPolinomialCoef5H2', 'UcomplexPolinomialCoef6H2', 'UcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.80      0.40      0.53        10
#            1       0.60      0.90      0.72        10

#     accuracy                           0.65        20
#    macro avg       0.70      0.65      0.63        20
# weighted avg       0.70      0.65      0.63        20


# Confusion Matrix:
#  [[4 6]
#  [1 9]]