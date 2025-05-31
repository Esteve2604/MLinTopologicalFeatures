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
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    #   * **n_neighbors:** The number of neighbors to consider when making a prediction.
    #   * A smaller value of `n_neighbors` makes the model more complex and can lead to overfitting (sensitive to noise).
    #   * A larger value of `n_neighbors` makes the model simpler and can lead to underfitting (less sensitive to patterns).
    #   * Odd values are often preferred in binary classification to avoid ties.

    'weights': ['uniform', 'distance'],
    #   * **weights:** Weight assigned to each neighbor.
    #   * 'uniform': All neighbors are weighted equally.
    #   * 'distance': Weight points by the inverse of their distance. Closer neighbors have a greater influence.
    #   * 'distance' weighting can be helpful when neighbors have varying degrees of relevance.

    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    #   * **algorithm:** Algorithm used to compute the nearest neighbors.
    #   * 'ball_tree': Uses BallTree.
    #   * 'kd_tree': Uses KDTree.
    #   * 'brute': Uses a brute-force search.
    #   * 'auto': Attempts to determine the most appropriate algorithm based on the values passed to fit method.
    #   * The best algorithm depends on the data size and structure.

    'p': [1, 2, 3],
    #   * **p:** Power parameter for the Minkowski metric.
    #   * 1: Manhattan distance.
    #   * 2: Euclidean distance.
    #   * Minkowski distance is a generalized distance metric.

    'leaf_size': [10, 20, 30, 40, 50]
    #   * **leaf_size:** Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
}

knn_classifier = KNeighborsClassifier()


# Use GridSearchCV with the custom scorer
grid_search = GridSearchCV(
    knn_classifier,
    param_grid,
    scoring='accuracy', 
    cv=CV,
    n_jobs=4,
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
# Best Parameters: {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 7, 'p': 1, 'weights': 'uniform'}
# Best Score: 0.6767857142857142
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.64      0.78      0.70         9
#            1       0.75      0.60      0.67        10

#     accuracy                           0.68        19
#    macro avg       0.69      0.69      0.68        19
# weighted avg       0.70      0.68      0.68        19


# Confusion Matrix:
#  [[7 2]
#  [4 6]]


# RESAMPLE 0.2
# Best Parameters: {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 3, 'p': 1, 'weights': 'distance'}
# Best Score: 0.9550000000000001
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.38      0.56      0.45         9
#            1       0.33      0.20      0.25        10

#     accuracy                           0.37        19
#    macro avg       0.36      0.38      0.35        19
# weighted avg       0.36      0.37      0.35        19


# Confusion Matrix:
#  [[5 4]
#  [8 2]]

# pDRed2.5EV8kHzStride2
# NO RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 15, 'p': 2, 'weights': 'uniform'}
# Best Score: 0.675
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

# RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 3, 'p': 1, 'weights': 'distance'}
# Best Score: 0.925
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
# Best Parameters: {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 9, 'p': 1, 'weights': 'uniform'}
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

# Only U
# NO RESAMPLE
# Best Parameters: {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 15, 'p': 1, 'weights': 'distance'}
# Best Score: 0.7375
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