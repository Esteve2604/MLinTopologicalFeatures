from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import  GridSearchCV
from imblearn.pipeline import Pipeline 

def trainModelWithSampling(X_train, y_train, X_test, y_test, preprocessor, sm, model, param_grid, cv, n_jobs):
    modelPipeline = Pipeline(steps=[
        ('samplingModel', sm),
        ('trainingModel', model)
    ])
    # Use GridSearchCV with the custom scorer
    grid_search = GridSearchCV(
        modelPipeline,
        param_grid,
        scoring='accuracy', 
        cv=cv,
        n_jobs=n_jobs,
        verbose=1
    )
    pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('gridSearch', grid_search)])
    pipeline.fit(X_train, y_train)

    #Extracting best score and parameters
    best_params = pipeline.named_steps['gridSearch'].best_params_
    best_score = pipeline.named_steps['gridSearch'].best_score_
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)
    best_pipeline =  pipeline.named_steps['gridSearch'].best_estimator_
    best_tree = best_pipeline.named_steps['trainingModel']

    #Training the best model obtained
    X_test = preprocessor.transform(X_test)
    y_pred = best_tree.predict(X_test)

    # Calculate evaluation metrics
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Print the metrics
    print("Validation Metrics:")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_matrix)