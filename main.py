from dataset_transform import dataset_transform
from algorithms import run_models
import pandas as pd


def main():
    X_train = pd.read_csv('pc_X_train.csv')
    y_train = pd.read_csv('pc_y_train.csv')
    validation = pd.read_csv('pc_X_test.csv')

    X_train, X_test, y_train, y_test, validation, validation_id = dataset_transform(
        X=X_train,
        y=y_train,
        validation=validation,
        test_size=0.3,
        random_state=42,
        preprocessing='StandardScaler',
        dim_reduction='PCA',
        dim_reduction_params={'n_components': 10}
    )

    dir_path = 'results/' + dataset_transform().get('preprocessing') + '_' + dataset_transform().get('dim_reduction')

    run_models(X_train, X_test, y_train, y_test, validation, validation_id)


if __name__ == "__main__":
    main()

