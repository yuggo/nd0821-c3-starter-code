from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200


def test_locally_post_poor():
    test_subject = {
    "age": 32,
    "workclass": "State-gov",
    "fnlgt": 0,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Exec-managerial",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Female",
    "capital_gain": 1000,
    "capital_loss": 20,
    "hours_per_week": 40,
    "native_country": "United-States"
    }

    r = client.post("/", json=test_subject)
    assert r.status_code == 200
    assert r.json() == "<=50K"


def test_locally_post_rich():
    test_subject = {
    "age": 42,
    "workclass": "Private",
    "fnlgt": 0,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 100000,
    "capital_loss": 20,
    "hours_per_week": 40,
    "native_country": "United-States"
    }

    r = client.post("/", json=test_subject)
    assert r.status_code == 200
    assert r.json() == ">50K"