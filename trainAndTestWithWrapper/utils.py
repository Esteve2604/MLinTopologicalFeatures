
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import  GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV, SequentialFeatureSelector

def selectFeaturesSFSAndRFECVAndTrainModel(X_train, y_train, X_test, y_test, preprocessor,  model, n_features_to_select, cv, n_jobs):
    
    rfecv = RFECV(
    estimator=model,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=1,
    n_jobs=n_jobs,
)
    pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('feature_selection', rfecv),
                            ('model', model)])
    pipeline.fit(X_train, y_train)

    selected_mask = pipeline.named_steps['feature_selection'].support_
    feature_names = X_train.columns
    selected_features = feature_names[selected_mask]
    features = list(selected_features)
    print("\n___RFECV____:\n")
    
    print("Nº of features selected:", len(features))
    
    print("Selected features:", features)

    #Training the best model obtained
    X_test_transformed = preprocessor.transform(X_test)
    X_test_selected = X_test_transformed[:, selected_mask]
    y_pred = pipeline.named_steps['model'].predict(X_test_selected)

    # Calculate evaluation metrics
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Print the metrics
    print("Validation Metrics:")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_matrix)
    
    selectFeaturesSFSAndTrainModel(X_train, y_train, X_test, y_test, preprocessor,  model, n_features_to_select, cv, n_jobs)
    
def selectFeaturesSFSAndTrainModel(X_train, y_train, X_test, y_test, preprocessor,  model, n_features_to_select, cv, n_jobs):
    
    sfs = SequentialFeatureSelector(
    estimator=model,
    n_features_to_select=n_features_to_select,
    direction="backward",
    cv=cv,
    scoring="accuracy",
    n_jobs=n_jobs,
)
    pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('feature_selection', sfs),
                            ('model', model)])
    pipeline.fit(X_train, y_train)

    selected_mask = pipeline.named_steps['feature_selection'].support_
    
    feature_names = X_train.columns
    selected_features = feature_names[selected_mask]
    features = list(selected_features)
    print("\n___SFS____:\n")
    
    print("Nº of features selected:", len(features))
    
    print("Selected features:", features)

    #Training the best model obtained
    X_test_transformed = preprocessor.transform(X_test)
    X_test_selected = X_test_transformed[:, selected_mask]
    y_pred = pipeline.named_steps['model'].predict(X_test_selected)

    # Calculate evaluation metrics
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Print the metrics
    print("Validation Metrics:")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_matrix)
    
