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
from utils import selectFeaturesSFSAndRFECVAndTrainModel

N_JOBS = 2
N_FEATURES_TO_SELECT = 15
CV = 2

# df = pd.read_csv("./data/featuresPDRed2.5EV8kHzStride2.csv")
# df = pd.read_csv("./data/featuresPDRed13EV41kHzStride2.csv")

datasetPath = "./data/afeaturesPDRed13EV41kHzStride2.csv"
datasetPath = "./data/efeaturesPDRed13EV41kHzStride2.csv"
datasetPath = "./data/ifeaturesPDRed13EV41kHzStride2.csv"
datasetPath = "./data/ofeaturesPDRed13EV41kHzStride2.csv"
datasetPath = "./data/ufeaturesPDRed13EV41kHzStride2.csv"

df = pd.read_csv(datasetPath)
print("El dataset utilizado es: " + datasetPath)

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
# sm = SVMSMOTE(sampling_strategy={0: 40, 1: 40}, m_neighbors=5, k_neighbors=4)
# X_train, y_train = sm.fit_resample(X_train,y_train)


model = DecisionTreeClassifier(random_state=42, class_weight=None, criterion="entropy", max_depth=None, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=2, min_samples_split=2, splitter='best')
selectFeaturesSFSAndRFECVAndTrainModel(X_train, y_train,X_test, y_test, preprocessor, model, N_FEATURES_TO_SELECT, CV, N_JOBS)

# ___RFECV____:

# Nº of features selected: 24
# Selected features: ['UcomplexPolinomialCoef7H0', 'UcomplexPolinomialCoef8H0', 'UcomplexPolinomialCoef9H0', 'UcomplexPolinomialCoef10H0', 'UcomplexPolinomialCoef1H1', 'UcomplexPolinomialCoef2H1', 'UcomplexPolinomialCoef3H1', 'UcomplexPolinomialCoef4H1', 'UcomplexPolinomialCoef5H1', 'UcomplexPolinomialCoef6H1', 'UcomplexPolinomialCoef7H1', 'UcomplexPolinomialCoef8H1', 'UcomplexPolinomialCoef9H1', 'UcomplexPolinomialCoef10H1', 'UcomplexPolinomialCoef1H2', 'UcomplexPolinomialCoef2H2', 'UcomplexPolinomialCoef3H2', 'UcomplexPolinomialCoef4H2', 'UcomplexPolinomialCoef5H2', 'UcomplexPolinomialCoef6H2', 'UcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.54      0.70      0.61        10
#            1       0.57      0.40      0.47        10

#     accuracy                           0.55        20
#    macro avg       0.55      0.55      0.54        20
# weighted avg       0.55      0.55      0.54        20


# Confusion Matrix:
#  [[7 3]
#  [6 4]]

# ___RFECV____:

# Nº of features selected: 170
# Selected features: ['AcomplexPolinomialCoef8H2', 'AcomplexPolinomialCoef10H2', 'EentropyH0', 'EentropyH1', 'EentropyH2', 'ElandscapeAmplitudeH0', 'ElandscapeAmplitudeH1', 'ElandscapeAmplitudeH2', 'EnumberOfPointsH0', 'EnumberOfPointsH1', 'EnumberOfPointsH2', 'EbettiAmplitudeH0', 'EbettiAmplitudeH1', 'EbettiAmplitudeH2', 'EcomplexPolinomialCoef1H0', 'EcomplexPolinomialCoef2H0', 'EcomplexPolinomialCoef3H0', 'EcomplexPolinomialCoef4H0', 'EcomplexPolinomialCoef5H0', 'EcomplexPolinomialCoef6H0', 'EcomplexPolinomialCoef7H0', 'EcomplexPolinomialCoef8H0', 'EcomplexPolinomialCoef9H0', 'EcomplexPolinomialCoef10H0', 'EcomplexPolinomialCoef1H1', 'EcomplexPolinomialCoef2H1', 'EcomplexPolinomialCoef3H1', 'EcomplexPolinomialCoef4H1', 'EcomplexPolinomialCoef5H1', 'EcomplexPolinomialCoef6H1', 'EcomplexPolinomialCoef7H1', 'EcomplexPolinomialCoef8H1', 'EcomplexPolinomialCoef9H1', 'EcomplexPolinomialCoef10H1', 'EcomplexPolinomialCoef1H2', 'EcomplexPolinomialCoef2H2', 'EcomplexPolinomialCoef3H2', 'EcomplexPolinomialCoef4H2', 'EcomplexPolinomialCoef5H2', 'EcomplexPolinomialCoef6H2', 'EcomplexPolinomialCoef7H2', 'EcomplexPolinomialCoef8H2', 'EcomplexPolinomialCoef9H2', 'EcomplexPolinomialCoef10H2', 'IentropyH0', 'IentropyH1', 'IentropyH2', 'IlandscapeAmplitudeH0', 'IlandscapeAmplitudeH1', 'IlandscapeAmplitudeH2', 'InumberOfPointsH0', 'InumberOfPointsH1', 'InumberOfPointsH2', 'IbettiAmplitudeH0', 'IbettiAmplitudeH1', 'IbettiAmplitudeH2', 'IcomplexPolinomialCoef1H0', 'IcomplexPolinomialCoef2H0', 'IcomplexPolinomialCoef3H0', 'IcomplexPolinomialCoef4H0', 'IcomplexPolinomialCoef5H0', 'IcomplexPolinomialCoef6H0', 'IcomplexPolinomialCoef7H0', 'IcomplexPolinomialCoef8H0', 'IcomplexPolinomialCoef9H0', 'IcomplexPolinomialCoef10H0', 'IcomplexPolinomialCoef1H1', 'IcomplexPolinomialCoef2H1', 'IcomplexPolinomialCoef3H1', 'IcomplexPolinomialCoef4H1', 'IcomplexPolinomialCoef5H1', 'IcomplexPolinomialCoef6H1', 'IcomplexPolinomialCoef7H1', 'IcomplexPolinomialCoef8H1', 'IcomplexPolinomialCoef9H1', 'IcomplexPolinomialCoef10H1', 'IcomplexPolinomialCoef1H2', 'IcomplexPolinomialCoef2H2', 'IcomplexPolinomialCoef3H2', 'IcomplexPolinomialCoef4H2', 'IcomplexPolinomialCoef5H2', 'IcomplexPolinomialCoef6H2', 'IcomplexPolinomialCoef7H2', 'IcomplexPolinomialCoef8H2', 'IcomplexPolinomialCoef9H2', 'IcomplexPolinomialCoef10H2', 'OentropyH0', 'OentropyH1', 'OentropyH2', 'OlandscapeAmplitudeH0', 'OlandscapeAmplitudeH1', 'OlandscapeAmplitudeH2', 'OnumberOfPointsH0', 'OnumberOfPointsH1', 'OnumberOfPointsH2', 'ObettiAmplitudeH0', 'ObettiAmplitudeH1', 'ObettiAmplitudeH2', 'OcomplexPolinomialCoef1H0', 'OcomplexPolinomialCoef2H0', 'OcomplexPolinomialCoef3H0', 'OcomplexPolinomialCoef4H0', 'OcomplexPolinomialCoef5H0', 'OcomplexPolinomialCoef6H0', 'OcomplexPolinomialCoef7H0', 'OcomplexPolinomialCoef8H0', 'OcomplexPolinomialCoef9H0', 'OcomplexPolinomialCoef10H0', 'OcomplexPolinomialCoef1H1', 'OcomplexPolinomialCoef2H1', 'OcomplexPolinomialCoef3H1', 'OcomplexPolinomialCoef4H1', 'OcomplexPolinomialCoef5H1', 'OcomplexPolinomialCoef6H1', 'OcomplexPolinomialCoef7H1', 'OcomplexPolinomialCoef8H1', 'OcomplexPolinomialCoef9H1', 'OcomplexPolinomialCoef10H1', 'OcomplexPolinomialCoef1H2', 'OcomplexPolinomialCoef2H2', 'OcomplexPolinomialCoef3H2', 'OcomplexPolinomialCoef4H2', 'OcomplexPolinomialCoef5H2', 'OcomplexPolinomialCoef6H2', 'OcomplexPolinomialCoef7H2', 'OcomplexPolinomialCoef8H2', 'OcomplexPolinomialCoef9H2', 'OcomplexPolinomialCoef10H2', 'UentropyH0', 'UentropyH1', 'UentropyH2', 'UlandscapeAmplitudeH0', 'UlandscapeAmplitudeH1', 'UlandscapeAmplitudeH2', 'UnumberOfPointsH0', 'UnumberOfPointsH1', 'UnumberOfPointsH2', 'UbettiAmplitudeH0', 'UbettiAmplitudeH1', 'UbettiAmplitudeH2', 'UcomplexPolinomialCoef1H0', 'UcomplexPolinomialCoef2H0', 'UcomplexPolinomialCoef3H0', 'UcomplexPolinomialCoef4H0', 'UcomplexPolinomialCoef5H0', 'UcomplexPolinomialCoef6H0', 'UcomplexPolinomialCoef7H0', 'UcomplexPolinomialCoef8H0', 'UcomplexPolinomialCoef9H0', 'UcomplexPolinomialCoef10H0', 'UcomplexPolinomialCoef1H1', 'UcomplexPolinomialCoef2H1', 'UcomplexPolinomialCoef3H1', 'UcomplexPolinomialCoef4H1', 'UcomplexPolinomialCoef5H1', 'UcomplexPolinomialCoef6H1', 'UcomplexPolinomialCoef7H1', 'UcomplexPolinomialCoef8H1', 'UcomplexPolinomialCoef9H1', 'UcomplexPolinomialCoef10H1', 'UcomplexPolinomialCoef1H2', 'UcomplexPolinomialCoef2H2', 'UcomplexPolinomialCoef3H2', 'UcomplexPolinomialCoef4H2', 'UcomplexPolinomialCoef5H2', 'UcomplexPolinomialCoef6H2', 'UcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.50      0.70      0.58        10
#            1       0.50      0.30      0.38        10

#     accuracy                           0.50        20
#    macro avg       0.50      0.50      0.48        20
# weighted avg       0.50      0.50      0.48        20


# Confusion Matrix:
#  [[7 3]
#  [7 3]]

# ___SFS____:

# Nº of features selected: 15
# Selected features: ['IcomplexPolinomialCoef7H1', 'IcomplexPolinomialCoef8H1', 'IcomplexPolinomialCoef9H1', 'IcomplexPolinomialCoef10H1', 'OcomplexPolinomialCoef8H1', 'OcomplexPolinomialCoef9H1', 'OcomplexPolinomialCoef6H2', 'OcomplexPolinomialCoef8H2', 'OcomplexPolinomialCoef9H2', 'OcomplexPolinomialCoef10H2', 'UcomplexPolinomialCoef5H2', 'UcomplexPolinomialCoef6H2', 'UcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef9H2']
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
# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2
# ___RFECV____:

# Nº of features selected: 65
# Selected features: ['OcomplexPolinomialCoef8H0', 'OcomplexPolinomialCoef9H0', 'OcomplexPolinomialCoef10H0', 'OcomplexPolinomialCoef1H1', 'OcomplexPolinomialCoef2H1', 'OcomplexPolinomialCoef3H1', 'OcomplexPolinomialCoef4H1', 'OcomplexPolinomialCoef5H1', 'OcomplexPolinomialCoef6H1', 'OcomplexPolinomialCoef7H1', 'OcomplexPolinomialCoef8H1', 'OcomplexPolinomialCoef9H1', 'OcomplexPolinomialCoef10H1', 'OcomplexPolinomialCoef1H2', 'OcomplexPolinomialCoef2H2', 'OcomplexPolinomialCoef3H2', 'OcomplexPolinomialCoef4H2', 'OcomplexPolinomialCoef5H2', 'OcomplexPolinomialCoef6H2', 'OcomplexPolinomialCoef7H2', 'OcomplexPolinomialCoef8H2', 'OcomplexPolinomialCoef9H2', 'OcomplexPolinomialCoef10H2', 'UentropyH0', 'UentropyH1', 'UentropyH2', 'UlandscapeAmplitudeH0', 'UlandscapeAmplitudeH1', 'UlandscapeAmplitudeH2', 'UnumberOfPointsH0', 'UnumberOfPointsH1', 'UnumberOfPointsH2', 'UbettiAmplitudeH0', 'UbettiAmplitudeH1', 'UbettiAmplitudeH2', 'UcomplexPolinomialCoef1H0', 'UcomplexPolinomialCoef2H0', 'UcomplexPolinomialCoef3H0', 'UcomplexPolinomialCoef4H0', 'UcomplexPolinomialCoef5H0', 'UcomplexPolinomialCoef6H0', 'UcomplexPolinomialCoef7H0', 'UcomplexPolinomialCoef8H0', 'UcomplexPolinomialCoef9H0', 'UcomplexPolinomialCoef10H0', 'UcomplexPolinomialCoef1H1', 'UcomplexPolinomialCoef2H1', 'UcomplexPolinomialCoef3H1', 'UcomplexPolinomialCoef4H1', 'UcomplexPolinomialCoef5H1', 'UcomplexPolinomialCoef6H1', 'UcomplexPolinomialCoef7H1', 'UcomplexPolinomialCoef8H1', 'UcomplexPolinomialCoef9H1', 'UcomplexPolinomialCoef10H1', 'UcomplexPolinomialCoef1H2', 'UcomplexPolinomialCoef2H2', 'UcomplexPolinomialCoef3H2', 'UcomplexPolinomialCoef4H2', 'UcomplexPolinomialCoef5H2', 'UcomplexPolinomialCoef6H2', 'UcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.50      0.40      0.44        10
#            1       0.50      0.60      0.55        10

#     accuracy                           0.50        20
#    macro avg       0.50      0.50      0.49        20
# weighted avg       0.50      0.50      0.49        20


# Confusion Matrix:
#  [[4 6]
#  [4 6]]

# ___SFS____:

# Nº of features selected: 15
# Selected features: ['EcomplexPolinomialCoef4H2', 'EcomplexPolinomialCoef5H2', 'EcomplexPolinomialCoef6H2', 'EcomplexPolinomialCoef7H2', 'EcomplexPolinomialCoef10H2', 'IcomplexPolinomialCoef3H1', 'IcomplexPolinomialCoef10H1', 'OcomplexPolinomialCoef3H2', 'OcomplexPolinomialCoef4H2', 'OcomplexPolinomialCoef5H2', 'OcomplexPolinomialCoef7H2', 'OcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef6H1', 'UcomplexPolinomialCoef7H1', 'UcomplexPolinomialCoef10H2']
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