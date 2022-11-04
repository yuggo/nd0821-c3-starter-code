import pytest
import pandas as pd
import pickle

@pytest.fixture
def data():
    """Simple function to get the sliced data"""
    PATH = "starter/data/slice_output.txt"
    df = pd.read_csv(PATH)
    return df

def test_available(data):
    """Test data is available."""
    assert data.shape[0] > 0

def test_results(data):
    """Test that only one slice at most performs bad"""
    assert (data["fbeta"] < 0.2).sum() <= 1