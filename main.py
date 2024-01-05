from dataset_transform import dataset_transform
from algorithms import run_models
import pandas as pd
import os


def main():
    X_train = pd.read_csv('pc_X_train.csv')
    y_train = pd.read_csv('pc_y_train.csv')
    validation = pd.read_csv('pc_X_test.csv')

    X_train, X_test, y_train, y_test, validation, validation_id, dir_path = dataset_transform(
        X=X_train,
        y=y_train,
        validation=validation,
        test_size=0.2,
        random_state=42,
        preprocessing='Normalizer', # StandardScaler, MinMaxScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer, PolynomialFeatures
    )

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    run_models(X_train, X_test, y_train, y_test, validation, validation_id, dir_path)


if __name__ == "__main__":
    main()

