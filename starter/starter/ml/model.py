from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    parameters = {
        "loss": ('log_loss', 'exponential'),
        "learning_rate": (0.001, 0.01, 0.1),
        "n_estimators": (100, 200),
        "subsample": (0.8, 0.9, 1),
        "max_depth": (3, 5)
    }

    model = GradientBoostingClassifier(random_state=0, max_features='auto')

    boost_grid = GridSearchCV(
        model, 
        param_grid = parameters,
        cv = 5, 
        verbose = False)
    
    boost_grid_model = boost_grid.fit(X_train, y_train)

    final_model = GradientBoostingClassifier(random_state=0, max_features='auto', **boost_grid_model.best_params_)
    final_model.fit(X_train)

    return final_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
