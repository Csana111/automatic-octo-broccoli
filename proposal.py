import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb

X = pd.read_csv('pc_X_train.csv')
y = pd.read_csv('pc_y_train.csv')

data = pd.merge(X, y, on='id')
data = data.drop(['id'], axis=1)

X = data.drop(['score'], axis=1)
y = data['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


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


# K-Nearest Neighbors
knn_model = KNeighborsRegressor()
knn_param_grid = {'n_neighbors': [14, 16, 18, 19]}
knn_best_model, knn_mse, knn_y_pred = perform_grid_search(knn_model, knn_param_grid, X_train, y_train, X_test, y_test,
                                                          'KNN')

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_param_grid = {'max_depth': [None, 4, 5, 6, 8], 'min_samples_split': [16, 17, 18], 'min_samples_leaf': [7, 8, 10]}
dt_best_model, dt_mse, dt_y_pred = perform_grid_search(dt_model, dt_param_grid, X_train, y_train, X_test, y_test,
                                                       'Decision Tree')

# Support Vector Machine
svm_model = SVR()
svm_param_grid = {'C': [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 1], 'kernel': ['linear', 'rbf']}
svm_best_model, svm_mse, svm_y_pred = perform_grid_search(svm_model, svm_param_grid, X_train, y_train, X_test, y_test,
                                                          'SVM')

# XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_param_grid = {'n_estimators': [90, 100, 150], 'max_depth': [None, 1, 2, 3, 4],
                  'learning_rate': [0.03, 0.05, 0.7]}
xgb_best_model, xgb_mse, xgb_y_pred = perform_grid_search(xgb_model, xgb_param_grid, X_train, y_train, X_test, y_test,
                                                          'XGBoost')

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_param_grid = {'n_estimators': [65, 100, 200], 'max_depth': [None, 10, 12, 15], 'min_samples_split': [2, 6, 8, 10],
                 'min_samples_leaf': [5, 7, 9]}
rf_best_model, rf_mse, rf_y_pred = perform_grid_search(rf_model, rf_param_grid, X_train, y_train, X_test, y_test,
                                                       'Random Forest')

# Ensemble
knn_y_pred = knn_best_model.predict(X_test)
dt_y_pred = dt_best_model.predict(X_test)
svm_y_pred = svm_best_model.predict(X_test)
xgb_y_pred = xgb_best_model.predict(X_test)
rf_y_pred = rf_best_model.predict(X_test)

ensemble_y_pred = (knn_y_pred + dt_y_pred + svm_y_pred + xgb_y_pred + rf_y_pred) / 5
ensemble_mse = mean_squared_error(y_test, ensemble_y_pred)
print(f"Mean Squared Error for Ensemble:", ensemble_mse)

validation = pd.read_csv('pc_X_test.csv')
validation_ids = validation['id']
validation = validation.drop(['id'], axis=1)
validation = scaler.transform(validation)

knn_y_pred = knn_best_model.predict(validation)
dt_y_pred = dt_best_model.predict(validation)
svm_y_pred = svm_best_model.predict(validation)
xgb_y_pred = xgb_best_model.predict(validation)
rf_y_pred = rf_best_model.predict(validation)

ensemble_y_pred = (knn_y_pred + dt_y_pred + svm_y_pred + xgb_y_pred + rf_y_pred) / 5

predictions = {
    'KNN': (knn_mse, knn_y_pred),
    'Decision Tree': (dt_mse, dt_y_pred),
    'SVM': (svm_mse, svm_y_pred),
    'XGBoost': (xgb_mse, xgb_y_pred),
    'Random Forest': (rf_mse, rf_y_pred)
}

for algo_name, (error, preds) in predictions.items():
    preds_df = pd.DataFrame()
    preds_df['id'] = validation_ids
    preds_df = preds_df.join(pd.DataFrame(preds, columns=['score']))
    preds_df.to_csv(f'{algo_name}_pred.csv', index=False)
    with open(f'{algo_name}_error.txt', 'w') as f:
        f.write(f"Mean Squared Error for {algo_name}: {error}")

end_preds_df = pd.DataFrame()
end_preds_df['id'] = validation_ids
ensemble_preds_df =  end_preds_df.join(pd.DataFrame(ensemble_y_pred, columns=['score']))
ensemble_preds_df.to_csv('ensemble_pred.csv', index=False)
with open('ensemble_error.txt', 'w') as f:
    f.write(f"Mean Squared Error for Ensemble: {ensemble_mse}")
