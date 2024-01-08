from dataset_transform import dataset_transform
from algorithms import run_models
from final import run_final
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
        preprocessing='PowerTransformer', # StandardScaler, MinMaxScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer, PolynomialFeatures
        preprocessing_params={},
        dim_reduction='SparsePCA',  # PCA, KernelPCA, SparsePCA, TruncatedSVD, FactorAnalysis
        dim_reduction_params={},
    )

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    run_final(X_train, X_test, y_train, y_test, validation, validation_id, dir_path)


if __name__ == "__main__":
    main()

