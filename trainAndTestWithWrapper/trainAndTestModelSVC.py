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
from utils import selectFeaturesSFSAndTrainModel

N_FEATURES_TO_SELECT = 15
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
# sm = SVMSMOTE(sampling_strategy={0: 100, 1: 100}, m_neighbors=5, k_neighbors=4)
# X_train, y_train = sm.fit_resample(X_train,y_train)

model = SVC(random_state=42, C=0.1, class_weight=None, coef0=1.0, degree=3, gamma='scale', kernel='poly', probability=False)
selectFeaturesSFSAndTrainModel(X_train, y_train,X_test, y_test, preprocessor, model, N_FEATURES_TO_SELECT, CV, N_JOBS)


# NO RESAMPLE 0.2
# ___SFS____:

# Nº of features selected: 15
# Selected features: ['IbettiAmplitudeH0', 'ObettiAmplitudeH1', 'OcomplexPolinomialCoef10H1', 'OcomplexPolinomialCoef9H2', 'OcomplexPolinomialCoef10H2', 'UentropyH1', 'UnumberOfPointsH1', 'UcomplexPolinomialCoef9H1', 'UcomplexPolinomialCoef3H2', 'UcomplexPolinomialCoef4H2', 'UcomplexPolinomialCoef5H2', 'UcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.64      0.90      0.75        10
#            1       0.83      0.50      0.62        10

#     accuracy                           0.70        20
#    macro avg       0.74      0.70      0.69        20
# weighted avg       0.74      0.70      0.69        20


# Confusion Matrix:
#  [[9 1]
#  [5 5]]
# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2
# ___SFS____:

# Nº of features selected: 15
# Selected features: ['IcomplexPolinomialCoef10H2', 'OnumberOfPointsH2', 'ObettiAmplitudeH1', 'UlandscapeAmplitudeH2', 'UnumberOfPointsH1', 'UbettiAmplitudeH1', 'UcomplexPolinomialCoef8H1', 'UcomplexPolinomialCoef9H1', 'UcomplexPolinomialCoef10H1', 'UcomplexPolinomialCoef2H2', 'UcomplexPolinomialCoef5H2', 'UcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.45      0.50      0.48        10
#            1       0.44      0.40      0.42        10

#     accuracy                           0.45        20
#    macro avg       0.45      0.45      0.45        20
# weighted avg       0.45      0.45      0.45        20


# Confusion Matrix:
#  [[5 5]
#  [6 4]]

# ___SFS____:

# Nº of features selected: 15
# Selected features: ['OentropyH2', 'OnumberOfPointsH1', 'OcomplexPolinomialCoef8H0', 'UlandscapeAmplitudeH1', 'UbettiAmplitudeH0', 'UbettiAmplitudeH1', 'UcomplexPolinomialCoef9H1', 'UcomplexPolinomialCoef10H1', 'UcomplexPolinomialCoef2H2', 'UcomplexPolinomialCoef5H2', 'UcomplexPolinomialCoef6H2', 'UcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef10H2']
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