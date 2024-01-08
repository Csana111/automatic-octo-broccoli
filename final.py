from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import xgboost as xgb
import numpy as np
import pandas as pd
import os

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


def run_final(X_train, X_test, y_train, y_test, validation, validation_ids, dir_path):
    # Support Vector Machine
    svm_model = SVR()
    svm_param_grid = {'C':  [0.7, 0.75, 0.8]}
    svm_best_model, svm_mse, svm_y_pred = perform_grid_search(svm_model, svm_param_grid, X_train, y_train, X_test,
                                                              y_test,
                                                              'SVM')

    # XGBoost
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_param_grid = {
        'n_estimators': [30, 35, 40, 50],
        'max_depth': [2, 3, 4],
        'subsample': [0.7, 0.8, 0.9, 1],
    }
    xgb_best_model, xgb_mse, xgb_y_pred = perform_grid_search(xgb_model, xgb_param_grid, X_train, y_train, X_test,
                                                              y_test,
                                                              'XGBoost')

    # Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_param_grid = {
        # estimators': [100, 200, 300, 500],
        'max_depth': [6, 8, 10, None],
        'min_samples_leaf': [3, 4, 5, 6, 7],
        # 'min_samples_split': [2, 5, 10]
    }
    rf_best_model, rf_mse, rf_y_pred = perform_grid_search(rf_model, rf_param_grid, X_train, y_train, X_test, y_test,
                                                           'Random Forest')
    # Ridge Regression
    ridge_regression_model = Ridge()
    ridge_param_grid = {'alpha': [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 1, 2, 4, 5, 6, 10]}
    ridge_best_model, ridge_mse, ridge_y_pred = perform_grid_search(ridge_regression_model, ridge_param_grid, X_train,
                                                                    y_train,
                                                                    X_test, y_test, 'Ridge Regression')
    # Models
    models = {
        'SVM': svm_best_model,
        'XGBoost': xgb_best_model,
        'Random Forest': rf_best_model,
        'Ridge Regression': ridge_best_model,
    }

    # Perform model selection
    model_errors = model_selection(models, X_train, y_train, X_test, y_test, error_metrics)

    # Save model errors
    model_errors_df = pd.DataFrame(model_errors)
    model_errors_df.to_csv(os.path.join(dir_path, 'model_errors.csv'), index=False)

    svm_y_pred = svm_best_model.predict(X_test)
    xgb_y_pred = xgb_best_model.predict(X_test)
    rf_y_pred = rf_best_model.predict(X_test)
    ridge_y_pred = ridge_best_model.predict(X_test)

    ensemble = (svm_y_pred + xgb_y_pred + rf_y_pred + ridge_y_pred) / 4

    for error in error_metrics:
        error_rate = error_metrics[error](y_test, ensemble)
        print(f"Ensemble {error}:", error_rate)

    best_models_3 = sorted(model_errors.items(), key=lambda x: x[1]['MSE'])[:3]
    best_models_3 = [model[0] for model in best_models_3]
    ensemble_y_pred_3 = np.zeros(len(y_test))
    for model_name in best_models_3:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        ensemble_y_pred_3 += y_pred / 3

    for error in error_metrics:
        error_rate = error_metrics[error](y_test, ensemble_y_pred_3)
        print(f"Ensemble_3 {error}:", error_rate)

    best_models_2 = sorted(model_errors.items(), key=lambda x: x[1]['MSE'])[:3]
    best_models_2 = [model[0] for model in best_models_2]
    ensemble_y_pred_2 = np.zeros(len(y_test))
    for model_name in best_models_2:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        ensemble_y_pred_2 += y_pred / 2

    for error in error_metrics:
        error_rate = error_metrics[error](y_test, ensemble_y_pred_2)
        print(f"Ensemble_2 {error}:", error_rate)

    end_preds_df = pd.DataFrame()
    end_preds_df['id'] = validation_ids

    svm_y_pred = svm_best_model.predict(validation)
    xgb_y_pred = xgb_best_model.predict(validation)
    rf_y_pred = rf_best_model.predict(validation)
    ridge_y_pred = ridge_best_model.predict(validation)
    ensemble = (svm_y_pred + xgb_y_pred + rf_y_pred + ridge_y_pred) / 4
    end_preds_df['score'] = ensemble
    end_preds_df.to_csv(os.path.join(dir_path, 'ensemble.csv'), index=False)

    ensemble_y_pred_3 = np.zeros(len(validation))
    for model_name in best_models_3:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(validation)
        ensemble_y_pred_3 += y_pred / 3
    end_preds_df['score'] = ensemble_y_pred_3
    end_preds_df.to_csv(os.path.join(dir_path, 'ensemble_3.csv'), index=False)

    ensemble_y_pred_2 = np.zeros(len(validation))
    for model_name in best_models_2:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(validation)
        ensemble_y_pred_2 += y_pred / 2
    end_preds_df['score'] = ensemble_y_pred_2
    end_preds_df.to_csv(os.path.join(dir_path, 'ensemble_2.csv'), index=False)

