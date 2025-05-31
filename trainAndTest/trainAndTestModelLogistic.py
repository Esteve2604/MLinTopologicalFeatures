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
 # Define the parameter grid for GridSearchCV
param_grid = {
    'penalty': ['elasticnet'],
    #   * **penalty:**
    #       * Specifies the norm used in the penalization.
    #       * 'l1': L1 regularization (Lasso). Promotes sparsity (feature selection).
    #       * 'l2': L2 regularization (Ridge). Shrinks coefficients.
    #       * 'elasticnet': Combination of L1 and L2.
    #       * 'none': No penalty.
    #       * Regularization helps prevent overfitting.

    'C': [0.001, 0.01, 0.1, 1, 10],
    #   * **C:**
    #       * Inverse of regularization strength.
    #       * Smaller values specify stronger regularization.
    #       * Larger values specify weaker regularization.
    #       * Tuning C is crucial for finding the right balance between bias and variance.

    'solver': ['saga'],
    #   * **solver:**
    #       * Algorithm to use in the optimization problem.
    #       * 'newton-cg', 'lbfgs', 'sag': Handle L2 penalty; some support multinomial.
    #       * 'liblinear': Handles L1 and L2; good for small datasets.
    #       * 'saga': Handles elasticnet; good for large datasets.
    #       * Solver choice is important for performance and whether it supports a given penalty.

    'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
    #   * **l1_ratio:**
    #       * Elastic-Net mixing parameter.
    #       * 0: L2 penalty.
    #       * 1: L1 penalty.
    #       * Between 0 and 1: Combination of L1 and L2.
    #       * Only used when `penalty='elasticnet'`.

    'class_weight': [None],
    #   * **class_weight:**
    #       * Weights associated with classes.
    #       * 'None': All classes are weighted equally.
    #       * 'balanced': Weights are inversely proportional to class frequencies.
    #       * Important for imbalanced datasets.

    'max_iter': [750, 1250, 2500, 5000, 7500, 10000],
    #   * **max_iter:**
    #       * Maximum number of iterations taken for the solvers to converge.
    #       * Increase if the solver is not converging.
}

tree_classifier = LogisticRegression(random_state=42)
# Use GridSearchCV with the custom scorer
grid_search = GridSearchCV(
    tree_classifier,
    param_grid,
    scoring='accuracy', 
    cv=5,
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

# NO RESAMPLE, 0.2
# Best Parameters: {'C': 0.1, 'class_weight': None, 'l1_ratio': 0.75, 'max_iter': 2500, 'penalty': 'elasticnet', 'solver': 'saga'}
# Best Score: 0.65
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.67      0.89      0.76         9
#            1       0.86      0.60      0.71        10

#     accuracy                           0.74        19
#    macro avg       0.76      0.74      0.73        19
# weighted avg       0.77      0.74      0.73        19


# Confusion Matrix:
#  [[8 1]
#  [4 6]]

# RESAMPLE, 0.2

# Best Parameters: {'C': 0.001, 'class_weight': None, 'l1_ratio': 0.0, 'max_iter': 5000, 'penalty': None, 'solver': 'saga'}
# Best Score: 0.845
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
# NO RESAMPLE 0.2 STANDARIZED - 5 folds
# Best Parameters: {'C': 1, 'class_weight': None, 'l1_ratio': 0.0, 'max_iter': 1250, 'penalty': 'elasticnet', 'solver': 'saga'}
# Best Score: 0.6625
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


# Confusion Matrix:
#  [[7 3]
#  [4 6]]
# RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'C': 10, 'class_weight': None, 'l1_ratio': 0.25, 'max_iter': 5000, 'penalty': 'elasticnet', 'solver': 'saga'}
# Best Score: 0.9299999999999999
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.58      0.70      0.64        10
#            1       0.62      0.50      0.56        10

#     accuracy                           0.60        20
#    macro avg       0.60      0.60      0.60        20
# weighted avg       0.60      0.60      0.60        20


# Confusion Matrix:
#  [[7 3]
#  [5 5]]

# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2

# Best Parameters: {'C': 0.01, 'class_weight': None, 'l1_ratio': 0.0, 'max_iter': 750, 'penalty': 'elasticnet', 'solver': 'saga'}
# Best Score: 0.7
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.69      0.90      0.78        10
#            1       0.86      0.60      0.71        10

#     accuracy                           0.75        20
#    macro avg       0.77      0.75      0.74        20
# weighted avg       0.77      0.75      0.74        20


# Confusion Matrix:
#  [[9 1]
#  [4 6]]

# ONLY U
# NO RESAMPLE
# Best Parameters: {'C': 1, 'class_weight': None, 'l1_ratio': 0.25, 'max_iter': 750, 'penalty': 'elasticnet', 'solver': 'saga'}
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