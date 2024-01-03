
# Load data
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

# Additional models
bayesian_ridge_model = BayesianRidge()
gaussian_nb_model = GaussianNB()
linear_regression_model = LinearRegression()
logistic_regression_model = LogisticRegression()
ridge_regression_model = Ridge()
lasso_regression_model = Lasso()
perceptron_model = Perceptron()
kmeans_model = KMeans(n_clusters=3, random_state=42)

# Model selection function
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

# Error metrics
error_metrics = {
    'MSE': mean_squared_error,
    'rMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'relative': lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    'relativeSE': lambda y_true, y_pred: np.mean(np.square((y_true - y_pred) / y_true)) * 100,
    'absoluteSE': mean_absolute_error,
    'statistical correlation': r2_score
}

# Models
models = {
    'Bayesian Ridge': bayesian_ridge_model,
    'Gaussian NB': gaussian_nb_model,
    'Linear Regression': linear_regression_model,
    'Logistic Regression': logistic_regression_model,
    'Ridge Regression': ridge_regression_model,
    'Lasso Regression': lasso_regression_model,
    'Perceptron': perceptron_model,
    'K-Means': kmeans_model
}

# Perform model selection
model_errors = model_selection(models, X_train, y_train, X_test, y_test, error_metrics)

# Print model errors
for model_name, errors in model_errors.items():
    print(f"{model_name}:")
    for error_name, error_value in errors.items():
        print(f"  {error_name}: {error_value}")
    print()