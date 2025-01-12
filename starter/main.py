# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

from starter.ml.data import process_data
from starter.ml.model import inference

import pickle
import pandas as pd


import os
import subprocess
import logging


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    dvc_output = subprocess.run(
        ["dvc", "pull"], capture_output=True, text=True)
    # print(dvc_output.stdout)
    # print(dvc_output.stderr)
    if dvc_output.returncode != 0:
        print("dvc pull failed")
        logging.warning("dvc pull failed")
    else:
        os.system("rm -r .dvc .apt/usr/lib/dvc")


example = {
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

example_full = {"example": example}

app = FastAPI()


class PredictItem(BaseModel):
    age: int
    workclass: Literal['State-gov',
                       'Self-emp-not-inc',
                       'Private',
                       'Federal-gov',
                       'Local-gov',
                       '?',
                       'Self-emp-inc',
                       'Without-pay',
                       'Never-worked']
    fnlgt: int
    education: Literal['Bachelors',
                       'HS-grad',
                       '11th',
                       'Masters',
                       '9th',
                       'Some-college',
                       'Assoc-acdm',
                       'Assoc-voc',
                       '7th-8th',
                       'Doctorate',
                       'Prof-school',
                       '5th-6th',
                       '10th',
                       '1st-4th',
                       'Preschool',
                       '12th']
    education_num: int
    marital_status: Literal['Never-married',
                            'Married-civ-spouse',
                            'Divorced',
                            'Married-spouse-absent',
                            'Separated',
                            'Married-AF-spouse',
                            'Widowed']
    occupation: Literal['Adm-clerical',
                        'Exec-managerial',
                        'Handlers-cleaners',
                        'Prof-specialty',
                        'Other-service',
                        'Sales',
                        'Craft-repair',
                        'Transport-moving',
                        'Farming-fishing',
                        'Machine-op-inspct',
                        'Tech-support',
                        '?',
                        'Protective-serv',
                        'Armed-Forces',
                        'Priv-house-serv']
    relationship: Literal['Not-in-family',
                          'Husband', 'Wife',
                          'Own-child', 'Unmarried',
                          'Other-relative']
    race: Literal['White', 'Black',
                  'Asian-Pac-Islander',
                  'Amer-Indian-Eskimo',
                  'Other']
    sex: Literal['Male', 'Female']
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal['United-States', 'Cuba',
                            'Jamaica', 'India', '?', 'Mexico',
                            'South', 'Puerto-Rico', 'Honduras',
                            'England', 'Canada', 'Germany',
                            'Iran', 'Philippines', 'Italy',
                            'Poland', 'Columbia', 'Cambodia',
                            'Thailand', 'Ecuador', 'Laos',
                            'Taiwan', 'Haiti', 'Portugal',
                            'Dominican-Republic',
                            'El-Salvador', 'France', 'Guatemala',
                            'China', 'Japan', 'Yugoslavia', 'Peru',
                            'Outlying-US(Guam-USVI-etc)',
                            'Scotland', 'Trinadad&Tobago',
                            'Greece', 'Nicaragua', 'Vietnam',
                            'Hong', 'Ireland', 'Hungary',
                            'Holand-Netherlands']

    class Config:
        schema_extra = example_full


PATH_TO_MODEL = "model.pkl"

with open(PATH_TO_MODEL, 'rb') as file:
    model = pickle.load(file)

PATH_TO_ENCODER = "encoder.pkl"

with open(PATH_TO_ENCODER, 'rb') as file:
    encoder = pickle.load(file)


@app.get("/")
async def root_hello():
    return "Hello World"


@app.post("/")
async def do_model_inference(item: PredictItem):
    # run model
    temp_dict = item.dict()
    final_dict = dict((key.replace('_', '-'), value)
                      for (key, value) in temp_dict.items())
    final_dict['salary'] = "NA"
    df = pd.DataFrame(final_dict, index=[0])

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
        df, categorical_features=cat_features,
        label="salary", training=False,
        encoder=encoder, lb=None
    )

    prediction = inference(model, X_test).tolist()[0]

    if prediction == 0:
        return "<=50K"

    elif prediction == 1:
        return ">50K"
