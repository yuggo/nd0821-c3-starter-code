# Script to train machine learning model.
# import libraries
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics_per_slice

# Add Path to Data
PATH_TO_FILE = "../data/census.csv"

# Add code to load in the data.
data = pd.read_csv(PATH_TO_FILE)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

PATH_TO_ENCODER = "../data/encoder.pkl"
with open(PATH_TO_ENCODER, 'wb') as file:
    pickle.dump(encoder, file)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

PATH_TO_MODEL = "../data/model.pkl"
with open(PATH_TO_MODEL, 'rb') as file:
    model = pickle.load(file)

per_slice = compute_model_metrics_per_slice(
    model, test, X_test, y_test, 'education'
)

OUTPUT_SLICE_PATH = "../data/slice_output.txt"

per_slice.to_csv(OUTPUT_SLICE_PATH, index=False, float_format='%.3f')

exit()

# Train and save a model.
model = train_model(X_train, y_train)

PATH_TO_MODEL = "../data/model.pkl"
with open(PATH_TO_MODEL, 'wb') as file:
    pickle.dump(model, file)
