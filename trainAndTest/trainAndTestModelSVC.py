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
    ])



X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y
    )
# Resample the training dataset to improve predictions
# sm = SVMSMOTE(sampling_strategy={0: 100, 1: 100}, m_neighbors=5, k_neighbors=4)
# X_train, y_train = sm.fit_resample(X_train,y_train)

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], #[0.001, 0.01, 0.1, 1, 10, 100]
    #   * **C:** Regularization parameter.
    #   * It controls the trade-off between achieving a low training error and a low testing error (generalization).
    #   * A smaller C leads to a wider margin but allows more training errors (more regularization).
    #   * A larger C leads to a narrower margin but aims to classify all training points correctly (less regularization).

    'kernel': ['linear', 'rbf','poly','sigmoid'],
    #   * **kernel:** Specifies the kernel type to be used in the algorithm.
    #   * 'linear': Linear kernel. Suitable for linearly separable data.
    #   * 'rbf': Radial Basis Function kernel. A very flexible kernel, often a good default choice for non-linear data.
    #   * 'poly': Polynomial kernel. Useful for polynomial relationships.
    #   * 'sigmoid': Sigmoid kernel. Can perform like a neural network in certain cases.

    'degree': [2,3],
    #   * **degree:** Degree of the polynomial kernel function ('poly'). Ignored by other kernels.
    #   * Controls the complexity of the polynomial curve.

    'gamma': ['scale', 'auto', 0.1, 1],
    #   * **gamma:** Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    #   * 'scale': Uses 1 / (n_features * X.var()) as value of gamma.
    #   * 'auto': Uses 1 / n_features.
    #   * Float: You can provide a specific float value.
    #   * 'gamma' defines the influence of a single training example.
    #   * A small gamma means a Gaussian with a large variance, so the influence of a data point is far-reaching.
    #   * A large gamma means a Gaussian with a small variance, so the influence of a data point is limited to its close neighborhood.

    'coef0': [0.0, 0.1, 0.5, 1.0],
    #   * **coef0:** Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
    #   * Influences the curve of the 'poly' and 'sigmoid' kernels.

    'class_weight': [None],
    #   * **class_weight:** Set the parameter C of class i to class_weight[i]*C for SVC.
    #   * 'None': All classes are treated equally.
    #   * 'balanced': Automatically adjust weights inversely proportional to class frequencies in the input data.
    #   * Important for imbalanced datasets.

    'probability': [False]
    #   * **probability:** Whether to enable probability estimates. This must be enabled prior to calling `fit`, and will slow down that method.
}

svc_classifier = SVC(random_state=42)

# Use GridSearchCV with the custom scorer
grid_search = GridSearchCV(
    svc_classifier,
    param_grid,
    scoring='accuracy', 
    cv=CV,
    n_jobs=-1,
    verbose=1
)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
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
# decisionTree = tree.DecisionTreeClassifier(class_weight={0:2,1:1})
# decisionTree.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = decisionTree.predict(X_test)

# Calculate evaluation metrics
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
# Print the metrics
print("Validation Metrics:")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

# Best Parameters: {'C': 0.1, 'class_weight': None, 'coef0': 0.0, 'degree': 2, 'gamma': 1, 'kernel': 'sigmoid', 'probability': False}
# Best Score: 0.6482142857142856
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

# Best Parameters: {'C': 1, 'class_weight': None, 'coef0': 0.0, 'degree': 2, 'gamma': 'scale', 'kernel': 'linear', 'probability': False}
# Best Score: 0.95
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

# pDRed2.5EV8kHzStride2
# NO RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'C': 0.01, 'class_weight': None, 'coef0': 1.0, 'degree': 3, 'gamma': 0.1, 'kernel': 'poly', 'probability': False}
# Best Score: 0.6875
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
# Best Parameters: {'C': 1, 'class_weight': None, 'coef0': 0.0, 'degree': 2, 'gamma': 'scale', 'kernel': 'linear', 'probability': False}
# Best Score: 0.65
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
# RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'C': 10, 'class_weight': None, 'coef0': 0.0, 'degree': 2, 'gamma': 'scale', 'kernel': 'linear', 'probability': False}
# Best Score: 0.9650000000000001
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.60      0.90      0.72        10
#            1       0.80      0.40      0.53        10

#     accuracy                           0.65        20
#    macro avg       0.70      0.65      0.63        20
# weighted avg       0.70      0.65      0.63        20


# Confusion Matrix:
#  [[9 1]
#  [6 4]]

# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2
# Best Parameters: {'C': 0.1, 'class_weight': None, 'coef0': 1.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly', 'probability': False}
# Best Score: 0.7
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
# Best Parameters: {'C': 1, 'class_weight': None, 'coef0': 0.0, 'degree': 2, 'gamma': 'scale', 'kernel': 'sigmoid', 'probability': False}
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