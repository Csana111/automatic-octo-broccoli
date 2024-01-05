import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor


# Error metrics
error_metrics = {
    'MSE': mean_squared_error,
    'rMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'relative': lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    'relativeSE': lambda y_true, y_pred: np.mean(np.square((y_true - y_pred) / y_true)) * 100,
    'absoluteSE': mean_absolute_error,
    'statistical correlation': r2_score
}


def perform_grid_search(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    grd = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                       verbose=2, n_jobs=-1)
    grd.fit(X_train, y_train)
    best = grd.best_params_
    print(f"Best Parameters for {model_name}:", best)

    model_ = model.set_params(**best)
    model_.fit(X_train, y_train)
    y_pred = model_.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for {model_name} (Best Model):", mse)

    return model_, mse, y_pred


def model_selection(models, X_train, y_train, X_test, y_test, error_metrics):
    model_errors = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        errors = {}
        for error_name, error_func in error_metrics.items():
            errors[error_name] = error_func(y_test, y_pred)

        model_errors[model_name] = errors

    return model_errors


def run_models(X_train, y_train, X_test, y_test, validation):
    # K-Nearest Neighbors
    knn_model = KNeighborsRegressor()
    knn_param_grid = {'n_neighbors': [5, 17, 18, 19], 'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': [5, 10, 15],
                      'p': [1, 2]}
    knn_best_model, knn_mse, knn_y_pred = perform_grid_search(knn_model, knn_param_grid, X_train, y_train, X_test,
                                                              y_test,
                                                              'KNN')

    # Decision Tree
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_param_grid = {'max_depth': [None, 4, 5, 6, 8], 'min_samples_split': [2, 14, 15, 16, 17],
                     'min_samples_leaf': [1, 8, 10, 12]}
    dt_best_model, dt_mse, dt_y_pred = perform_grid_search(dt_model, dt_param_grid, X_train, y_train, X_test, y_test,
                                                           'Decision Tree')

    # Support Vector Machine
    svm_model = SVR()
    svm_param_grid = {'C': [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 1], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
    svm_best_model, svm_mse, svm_y_pred = perform_grid_search(svm_model, svm_param_grid, X_train, y_train, X_test,
                                                              y_test,
                                                              'SVM')

    # XGBoost
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_param_grid = {'n_estimators': [65, 70, 75, 80, 100, 200], 'max_depth': [None, 2, 3, 6],
                      'subsample': [0.8, 0.9, 1], 'colsample_bytree': [0.5, 0.6, 0.7, 1]}
    xgb_best_model, xgb_mse, xgb_y_pred = perform_grid_search(xgb_model, xgb_param_grid, X_train, y_train, X_test,
                                                              y_test,
                                                              'XGBoost')

    # Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_param_grid = {'n_estimators': [50, 55, 60],
                     'max_depth': [11, 12, 13, None], 'min_samples_leaf': [4, 5, 6], 'min_samples_split': [11, 12, 13]}
    rf_best_model, rf_mse, rf_y_pred = perform_grid_search(rf_model, rf_param_grid, X_train, y_train, X_test, y_test,
                                                           'Random Forest')

    # AdaBoost
    ada_model = AdaBoostRegressor(random_state=42)
    ada_param_grid = {'n_estimators': [10, 50, 100, 500], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]}
    ada_best_model, ada_mse, ada_y_pred = perform_grid_search(ada_model, ada_param_grid, X_train, y_train, X_test,
                                                              y_test,
                                                              'AdaBoost')

    # Bayesian Ridge
    bayesian_ridge_model = BayesianRidge()
    bay_param_grid = {'max_iter': [50, 100, 200, 300, 400, 500], 'tol': [0.01, 0.002, 1e-3, 1e-4, 1e-5, 1e-6]}
    bay_best_model, bay_mse, bay_y_pred = perform_grid_search(bayesian_ridge_model, bay_param_grid, X_train, y_train,

                                                              X_test, y_test, 'Bayesian Ridge')

    # Linear Regression
    linear_regression_model = LinearRegression()
    linear_param_grid = {'fit_intercept': [True, False], 'copy_X': [True, False]}
    linear_best_model, linear_mse, linear_y_pred = perform_grid_search(linear_regression_model, linear_param_grid,
                                                                       X_train,
                                                                       y_train,
                                                                       X_test, y_test, 'Linear Regression')

    # Ridge Regression
    ridge_regression_model = Ridge()
    ridge_param_grid = {'alpha': [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 1, 2, 5], 'fit_intercept': [True, False],
                        'copy_X': [True, False]}
    ridge_best_model, ridge_mse, ridge_y_pred = perform_grid_search(ridge_regression_model, ridge_param_grid, X_train,
                                                                    y_train,
                                                                    X_test, y_test, 'Ridge Regression')

    # Lasso Regression
    lasso_regression_model = Lasso()
    lasso_param_grid = {'alpha': [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 1], 'fit_intercept': [True, False],
                        'copy_X': [True, False]}
    lasso_best_model, lasso_mse, lasso_y_pred = perform_grid_search(lasso_regression_model, lasso_param_grid, X_train,
                                                                    y_train,
                                                                    X_test, y_test, 'Lasso Regression')

    # K-Means
    kmeans_model = KMeans()
    kmeans_param_grid = {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]}
    kmeans_best_model, kmeans_mse, kmeans_y_pred = perform_grid_search(kmeans_model, kmeans_param_grid, X_train,
                                                                       y_train,
                                                                       X_test, y_test, 'K-Means')

    # Neural Network
    nn_model = MLPRegressor()
    nn_param_grid = {'hidden_layer_sizes': [(100,), (500,), (100, 100), (100, 500), (500, 100), (100, 100, 100)],
                     'activation': ['relu', 'tanh', 'logistic'], 'solver': ['adam', 'sgd'],
                     'learning_rate': ['constant', 'invscaling', 'adaptive']}
    nn_best_model, nn_mse, nn_y_pred = perform_grid_search(nn_model, nn_param_grid, X_train, y_train, X_test, y_test,
                                                           'Neural Network')

    # Models
    models = {
        'KNN': knn_best_model,
        'Decision Tree': dt_best_model,
        'SVM': svm_best_model,
        'XGBoost': xgb_best_model,
        'Random Forest': rf_best_model,
        'AdaBoost': ada_best_model,
        'Bayesian Ridge': bay_best_model,
        'Linear Regression': linear_best_model,
        'Ridge Regression': ridge_best_model,
        'Lasso Regression': lasso_best_model,
        'K-Means': kmeans_best_model,
        'Neural Network': nn_best_model
    }

    # Perform model selection
    model_errors = model_selection(models, X_train, y_train, X_test, y_test, error_metrics)

    # Print model errors
    for model_name, errors in model_errors.items():
        print(f"{model_name}:")
        for error_name, error_value in errors.items():
            print(f"  {error_name}: {error_value}")
        print()

    # Ensemble all models
    knn_y_pred = knn_best_model.predict(X_test)
    dt_y_pred = dt_best_model.predict(X_test)
    svm_y_pred = svm_best_model.predict(X_test)
    xgb_y_pred = xgb_best_model.predict(X_test)
    rf_y_pred = rf_best_model.predict(X_test)
    ada_y_pred = ada_best_model.predict(X_test)
    bay_y_pred = bay_best_model.predict(X_test)
    linear_y_pred = linear_best_model.predict(X_test)
    ridge_y_pred = ridge_best_model.predict(X_test)
    lasso_y_pred = lasso_best_model.predict(X_test)
    kmeans_y_pred = kmeans_best_model.predict(X_test)
    neural_y_pred = nn_best_model.predict(X_test)

    ensemble_y_pred = (
                              knn_y_pred + dt_y_pred + svm_y_pred + xgb_y_pred + rf_y_pred + ada_y_pred + bay_y_pred + linear_y_pred + ridge_y_pred + lasso_y_pred + kmeans_y_pred + neural_y_pred) / 12

    for error in error_metrics:
        error_rate = error_metrics[error](y_test, ensemble_y_pred)
        print(f"Ensemble {error}:", error_rate)

    best_models = sorted(model_errors.items(), key=lambda x: x[1]['MSE'])[:5]
    best_models = [model[0] for model in best_models]

    ensemble_y_pred_best = np.zeros(len(y_test))
    y_pred = np.zeros(len(y_test))
    for model_name in best_models:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    ensemble_y_pred_best += y_pred / 5

    for error in error_metrics:
        error_rate = error_metrics[error](y_test, ensemble_y_pred_best)
        print(f"Ensemble {error}:", error_rate)

    validation_ids = validation['id']
    validation = validation.drop(['id'], axis=1)

    knn_y_pred = knn_best_model.predict(validation)
    dt_y_pred = dt_best_model.predict(validation)
    svm_y_pred = svm_best_model.predict(validation)
    xgb_y_pred = xgb_best_model.predict(validation)
    rf_y_pred = rf_best_model.predict(validation)
    ada_y_pred = ada_best_model.predict(validation)
    bay_y_pred = bay_best_model.predict(validation)
    linear_y_pred = linear_best_model.predict(validation)
    ridge_y_pred = ridge_best_model.predict(validation)
    lasso_y_pred = lasso_best_model.predict(validation)
    kmeans_y_pred = kmeans_best_model.predict(validation)
    neual_y_pred = nn_best_model.predict(validation)
    ensemble_y_pred = (
                              knn_y_pred + dt_y_pred + svm_y_pred + xgb_y_pred + rf_y_pred + ada_y_pred + bay_y_pred + linear_y_pred + ridge_y_pred + lasso_y_pred + kmeans_y_pred + neual_y_pred) / 12

    ensemble_y_pred_best = np.zeros(len(validation))
    y_pred = np.zeros(len(validation))
    for model_name in best_models:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(validation)
    ensemble_y_pred_best += y_pred / 5

    predictions = {
        'KNN': (knn_mse, knn_y_pred),
        'Decision Tree': (dt_mse, dt_y_pred),
        'SVM': (svm_mse, svm_y_pred),
        'XGBoost': (xgb_mse, xgb_y_pred),
        'Random Forest': (rf_mse, rf_y_pred),
        'AdaBoost': (ada_mse, ada_y_pred),
        'Bayesian Ridge': (bay_mse, bay_y_pred),
        'Linear Regression': (linear_mse, linear_y_pred),
        'Ridge Regression': (ridge_mse, ridge_y_pred),
        'Lasso Regression': (lasso_mse, lasso_y_pred),
        'K-Means': (kmeans_mse, kmeans_y_pred),
        'Neural Network': (nn_mse, neural_y_pred)
    }

    for algo_name, (errors, preds) in predictions.items():
        preds_df = pd.DataFrame()
        preds_df['id'] = validation_ids
        preds_df = preds_df.join(pd.DataFrame(preds, columns=['score']))
        preds_df.to_csv(f'{algo_name}_pred.csv', index=False)
        with open(f'{algo_name}_error.txt', 'w') as f:
            for model_name, errors in model_errors.items():
                for error_name, error_value in errors.items():
                    f.write(f"{error_name} for {algo_name}: {error_value}\n")

    end_preds_df = pd.DataFrame()
    end_preds_df['id'] = validation_ids
    ensemble_preds_df = end_preds_df.join(pd.DataFrame(ensemble_y_pred, columns=['score']))
    ensemble_preds_df.to_csv('ensemble_pred.csv', index=False)
    ensemble_preds_best_df = end_preds_df.join(pd.DataFrame(ensemble_y_pred_best, columns=['score']))
    ensemble_preds_best_df.to_csv('ensemble_best_pred.csv', index=False)
    with open('ensemble_error.txt', 'w') as f:
        for error in error_metrics:
            error_rate = error_metrics[error](y_test, ensemble_y_pred)
            f.write(f"Ensemble {error}: {error_rate}\n")

