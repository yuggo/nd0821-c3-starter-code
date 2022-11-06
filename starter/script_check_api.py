import requests
import json


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


response = requests.post('https://mlpipeline-heroku.herokuapp.com/',
                         data=json.dumps(test_subject))

print(response.status_code)
print(response.json())
