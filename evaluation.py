# evaluation.py

from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(model, x_cv, y_cv):
    predictions = model.predict(x_cv)
    accuracy = accuracy_score(y_cv, predictions)
    matrix = confusion_matrix(y_cv, predictions)
    return accuracy, matrix
