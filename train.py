# train

from config import TRAIN_PATH, TEST_PATH, OUTPUT_PATH, TEST_SIZE
from data_loader import load_data
from preprocessing import clean_data, prepare_features
from modeling import get_models
from evaluation import evaluate_model

from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
train, test = load_data(TRAIN_PATH, TEST_PATH)

# Clean data
train = clean_data(train)
test = clean_data(test)

# Prepare features
X, y, test = prepare_features(train, test)

# Train/validation split
x_train, x_cv, y_train, y_cv = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)

# Train & evaluate models
models = get_models()
results = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    acc, matrix = evaluate_model(model, x_cv, y_cv)
    results[name] = acc
    print(f"{name}: {acc:.2%}")
    print(matrix, "\n")

# Best model: Naive Bayes
best_model = models["Naive Bayes"]
best_model.fit(X, y)

# Predict test data
pred_test = best_model.predict(test)

# Save predictions
pd.DataFrame(pred_test, columns=["predictions"]).to_csv(OUTPUT_PATH, index=False)
