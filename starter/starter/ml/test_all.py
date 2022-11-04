import pytest
import pandas as pd
import pickle

from starter.ml.data import process_data
from starter.ml.model import inference


@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""
    df = pd.DataFrame(
        {
            "age": [39, 50, 38],
            "workclass": ["State-gov", "Self-emp-not-inc", "Private"],
            "fnlgt": [77516, 83311, 215646],
            "education": ["Bachelors", "Bachelors", "HS-grad"],
            "education-num": [13, 13, 9],
            "marital-status": ["Never-married",
                               "Married-civ-spouse", "Divorced"],
            "occupation": ["Adm-clerical",
                           "Exec-managerial", "Handlers-cleaners"],
            "relationship": ["Not-in-family",
                             "Husband", "Not-in-family"],
            "race": ["White", "White", "White"],
            "sex": ["Male", "Male", "Male"],
            "capital-gain": [2174, 0, 0],
            "capital-loss": [0, 0, 0],
            "hours-per-week": [40, 13, 40],
            "native-country": ["United-States",
                               "United-States", "United-States"],
            "salary": ["<=50K", "<=50K", "<=50K"]
        }
    )
    return df


@pytest.fixture
def model():
    """ Simple model load"""
    PATH_TO_MODEL = "data/model.pkl"

    with open(PATH_TO_MODEL, 'rb') as file:
        model = pickle.load(file)

    return model


@pytest.fixture
def encoder():
    """ Simple encoder load"""
    PATH_TO_ENCODER = "data/encoder.pkl"

    with open(PATH_TO_ENCODER, 'rb') as file:
        encoder = pickle.load(file)

    return encoder


def test_data_shape(data):
    assert data.shape[1] == 15


def test_model_present(model):
    assert model


def test_encoder_present(encoder):
    assert encoder


def test_inference(data, model, encoder):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_test, y_test, _, _ = process_data(
        data, categorical_features=cat_features,
        label="salary", training=False,
        encoder=encoder, lb=None
    )

    predictions = inference(model, X_test)

    assert predictions.shape[0] == 3
