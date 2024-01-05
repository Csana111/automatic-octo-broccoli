import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FactorAnalysis

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split


def dataset_transform(X, y, validation, test_size=0.2, random_state=42, preprocessing='StandardScaler', dim_reduction=None,
                      dim_reduction_params=None):

    data = pd.merge(X, y, on='id')
    data = data.drop(['id'], axis=1)

    validation_id = validation['id']
    validation = validation.drop(['id'], axis=1)

    train = data.drop(['score'], axis=1)
    y_ = data['score']

    X_train, X_test, y_train, y_test = train_test_split(train, y_, test_size=test_size, random_state=random_state)

    preprocessing_steps = []
    scaler = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'Normalizer': Normalizer(),
        'QuantileTransformer': QuantileTransformer(),
        'PowerTransformer': PowerTransformer(),
        'PolynomialFeatures': PolynomialFeatures(),
    }.get(preprocessing)
    if scaler:
        preprocessing_steps.append(('scaler', scaler))

    dim_reduction_params = dim_reduction_params or {}
    reducer = {
        'PCA': PCA(**dim_reduction_params),
        'KernelPCA': KernelPCA(**dim_reduction_params),
        'SparsePCA': SparsePCA(**dim_reduction_params),
        'TruncatedSVD': TruncatedSVD(**dim_reduction_params),
        'FactorAnalysis': FactorAnalysis(**dim_reduction_params),
    }.get(dim_reduction)
    if reducer:
        preprocessing_steps.append(('reducer', reducer))

    if preprocessing_steps:
        pipeline = Pipeline(steps=preprocessing_steps)
        X_train = pipeline.fit_transform(X_train)
        X_test = pipeline.transform(X_test)
        validation = pipeline.transform(validation)
    else:
        raise ValueError("No valid preprocessing or dimensionality reduction method provided")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test, validation, validation_id
