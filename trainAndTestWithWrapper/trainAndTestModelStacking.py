from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.svm import SVC

import matplotlib
from imblearn.over_sampling import SVMSMOTE
from utils import selectFeaturesSFSAndTrainModel
N_FEATURES_TO_SELECT = 15
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

# With resampling
# adaBoost =  AdaBoostClassifier(random_state=42, estimator=DecisionTreeClassifier(max_depth=3), learning_rate=1.0, n_estimators=100)
# logistic =  LogisticRegression(random_state=42,  class_weight=None, max_iter = 5000, penalty=None, solver='saga')
# rf = RandomForestClassifier(ccp_alpha=0.0, class_weight='balanced_subsample', criterion='gini', max_depth=None, max_features='log2', min_samples_leaf=1, min_samples_split=2, n_estimators=200)
# svc = SVC(C=1, class_weight=None, coef0=0.0, degree=2, gamma='scale', kernel='linear', probability=False)

# Without resampling 
adaBoost =  AdaBoostClassifier(random_state=42, estimator=DecisionTreeClassifier(max_depth=3), learning_rate=0.1, n_estimators=75)
logistic =  LogisticRegression(random_state=42,C=0.1, class_weight=None,penalty='elasticnet', l1_ratio=0.75, max_iter = 2500, solver='saga')
rf = RandomForestClassifier(ccp_alpha=0.05, class_weight='balanced_subsample', criterion='entropy', max_depth=None, max_features='log2', min_samples_leaf=2, min_samples_split=5, n_estimators=50)
svc = HistGradientBoostingClassifier(class_weight=None, early_stopping=True, l2_regularization=0.0, learning_rate=0.01, loss='log_loss', max_depth=None, max_iter=50, min_samples_leaf=20, n_iter_no_change=10, validation_fraction=0.1)

param_grid = [
    {
        'estimators': [
                    # 2-model combinations
                    [('rf', rf)],
                    [('adaBoost', adaBoost)],
                    [('svc', svc)],
                    [('logistic', logistic)],
                    [('rf', rf), ('adaBoost', adaBoost)],
                    [('rf', rf), ('svc', svc)],
                    [('rf', rf), ('logistic', logistic)],
                    [('adaBoost', adaBoost), ('svc', svc)],
                    [('adaBoost', adaBoost), ('logistic', logistic)],
                    [('svc', svc), ('logistic', logistic)],
    
                    # 3-model combinations
                    [('rf', rf), ('adaBoost', adaBoost), ('svc', svc)],
                    [('rf', rf), ('adaBoost', adaBoost), ('logistic', logistic)],
                    [('rf', rf), ('svc', svc), ('logistic', logistic)],
                    [('adaBoost', adaBoost), ('svc', svc), ('logistic', logistic)],
    
                    # 4-model combination
                    [('rf', rf), ('adaBoost', adaBoost), ('svc', svc), ('logistic', logistic)]

       
    ],
    'final_estimator': [
        DecisionTreeClassifier()
    ],
    'passthrough': [True, False]
    },
    {
   'estimators': [
                    # 2-model combinations
                    [('rf', rf)],
                    [('adaBoost', adaBoost)],
                    [('svc', svc)],
                    [('logistic', logistic)],
                    [('rf', rf), ('adaBoost', adaBoost)],
                    [('rf', rf), ('svc', svc)],
                    [('rf', rf), ('logistic', logistic)],
                    [('adaBoost', adaBoost), ('svc', svc)],
                    [('adaBoost', adaBoost), ('logistic', logistic)],
                    [('svc', svc), ('logistic', logistic)],
    
                    # 3-model combinations
                    [('rf', rf), ('adaBoost', adaBoost), ('svc', svc)],
                    [('rf', rf), ('adaBoost', adaBoost), ('logistic', logistic)],
                    [('rf', rf), ('svc', svc), ('logistic', logistic)],
                    [('adaBoost', adaBoost), ('svc', svc), ('logistic', logistic)],
    
                    # 4-model combination
                    [('rf', rf), ('adaBoost', adaBoost), ('svc', svc), ('logistic', logistic)]

       
    ],
    'final_estimator': [
        LogisticRegression(solver='saga', class_weight=None),
    ], #'final_estimator__C': 1, 'final_estimator__l1_ratio': 0.25, 'final_estimator__max_iter': 1000, 'final_estimator__penalty': 'elasticnet'
    'final_estimator__C': [1],
    'final_estimator__penalty': ['elasticnet'],
    'final_estimator__l1_ratio': [0.25],
    'final_estimator__max_iter': [1000],

    'passthrough': [True, False]
    },
    {
      'estimators': [
                    # 2-model combinations
                    [('rf', rf)],
                    [('adaBoost', adaBoost)],
                    [('svc', svc)],
                    [('logistic', logistic)],
                    [('rf', rf), ('adaBoost', adaBoost)],
                    [('rf', rf), ('svc', svc)],
                    [('rf', rf), ('logistic', logistic)],
                    [('adaBoost', adaBoost), ('svc', svc)],
                    [('adaBoost', adaBoost), ('logistic', logistic)],
                    [('svc', svc), ('logistic', logistic)],
    
                    # 3-model combinations
                    [('rf', rf), ('adaBoost', adaBoost), ('svc', svc)],
                    [('rf', rf), ('adaBoost', adaBoost), ('logistic', logistic)],
                    [('rf', rf), ('svc', svc), ('logistic', logistic)],
                    [('adaBoost', adaBoost), ('svc', svc), ('logistic', logistic)],
    
                    # 4-model combination
                    [('rf', rf), ('adaBoost', adaBoost), ('svc', svc), ('logistic', logistic)]

       
    ],
    'final_estimator': [
       MultinomialNB(),
    ],
    'final_estimator__alpha': [0.1, 1.0, 10.0],  # Tune alpha
    'passthrough': [True, False]
    }
]


for param_grid in param_grid:
    # Use GridSearchCV with the custom scorer
    grid_search = GridSearchCV(
        StackingClassifier(estimators=[]),
        param_grid,
        scoring='accuracy', 
        cv=CV,
        n_jobs=-1,
        verbose=0
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
    X_test_prep = preprocessor.transform(X_test)
    y_pred = best_tree.predict(X_test_prep)
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

# NO RESAMPLING 0.2
# Decision Tree
# Best Parameters: {'estimators': [('svc', HistGradientBoostingClassifier(early_stopping=True, learning_rate=0.01,
#                                max_iter=50))], 'final_estimator': DecisionTreeClassifier(), 'passthrough': True}
# Best Score: 0.7017857142857142
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.67      0.67      0.67         9
#            1       0.70      0.70      0.70        10

#     accuracy                           0.68        19
#    macro avg       0.68      0.68      0.68        19
# weighted avg       0.68      0.68      0.68        19


# Confusion Matrix:
#  [[6 3]
#  [3 7]]

# LOGISTIC
# Best Parameters: {'estimators': [('rf', RandomForestClassifier(ccp_alpha=0.05, class_weight='balanced_subsample',
#                        criterion='entropy', max_features='log2',
#                        min_samples_leaf=4, n_estimators=500))], 'final_estimator': LogisticRegression(solver='saga'), 'final_estimator__C': 1, 'final_estimator__l1_ratio': 0.25, 'final_estimator__max_iter': 1000, 'final_estimator__penalty': 'elasticnet', 'passthrough': False}
# Best Score: 0.6660714285714285
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       1.00      0.67      0.80         9
#            1       0.77      1.00      0.87        10

#     accuracy                           0.84        19
#    macro avg       0.88      0.83      0.83        19
# weighted avg       0.88      0.84      0.84        19

# NAIVE BAYES
# Best Parameters: {'estimators': [('rf', RandomForestClassifier(ccp_alpha=0.05, class_weight='balanced_subsample',
#                        criterion='entropy', max_features='log2',
#                        min_samples_leaf=4, n_estimators=500)), ('svc', HistGradientBoostingClassifier(early_stopping=True, learning_rate=0.01,
#                                max_iter=50)), ('logistic', LogisticRegression(C=0.1, l1_ratio=0.75, max_iter=2500, penalty='elasticnet',
#                    random_state=42, solver='saga'))], 'final_estimator': MultinomialNB(), 'final_estimator__alpha': 0.1, 'passthrough': False}
# Best Score: 0.6642857142857144
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       1.00      0.56      0.71         9
#            1       0.71      1.00      0.83        10

#     accuracy                           0.79        19
#    macro avg       0.86      0.78      0.77        19
# weighted avg       0.85      0.79      0.78        19


# Confusion Matrix:
#  [[ 5  4]
#  [ 0 10]]

# Confusion Matrix:
#  [[ 6  3]
#  [ 0 10]]

# RESAMPLING 0.2
# Decision Tree Classifier
# Best Parameters: {'estimators': [('adaBoost', AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3),
                #    n_estimators=150, random_state=42)), ('svc', SVC(C=1, degree=2, kernel='linear')), 
                # ('logistic', LogisticRegression(max_iter=5000, penalty=None, random_state=42, solver='saga'))], 
                # 'final_estimator': DecisionTreeClassifier(), 'passthrough': False}
# Best Score: 0.93
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.54      0.78      0.64         9
#            1       0.67      0.40      0.50        10

#     accuracy                           0.58        19
#    macro avg       0.60      0.59      0.57        19
# weighted avg       0.61      0.58      0.56        19


# Confusion Matrix:
#  [[7 2]
#  [6 4]]

# Logistic 
# Best Parameters: {'estimators': [('rf', RandomForestClassifier(class_weight='balanced_subsample', max_features='log2',
#                        n_estimators=200)), ('adaBoost', AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3),
#                    n_estimators=150, random_state=42)), ('svc', SVC(C=1, degree=2, kernel='linear')), 
#                   ('logistic', LogisticRegression(max_iter=5000, penalty=None, random_state=42, solver='saga'))], 'final_estimator': LogisticRegression(solver='saga'), 
#                   'final_estimator__C': 1, 'final_estimator__l1_ratio': 0.25, 'final_estimator__max_iter': 1000, 'final_estimator__penalty': 'elasticnet', 'passthrough': False}
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

# Naive Bayes
# Best Parameters: {'estimators': [('adaBoost', AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3),
#                    n_estimators=150, random_state=42)), 
#               ('logistic', LogisticRegression(max_iter=5000, penalty=None, random_state=42, solver='saga'))], 
#               'final_estimator': MultinomialNB(), 'final_estimator__alpha': 10.0, 'passthrough': False}
# Best Score: 0.9
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

# RESAMPLE 0.2 STANDARIZED