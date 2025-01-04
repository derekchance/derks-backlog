from pathlib import Path
from os import path
import joblib
from sklearn.ensemble import StackingRegressor

from .core import MODEL_DIR, load_Xy


def main():
    X, y = load_Xy()
    xgb_model = joblib.load(MODEL_DIR / './models/xgb_model1.joblib')
    xgb_model2 = joblib.load(MODEL_DIR / './models/xgb_model2.joblib')
    xgb_model3 = joblib.load(MODEL_DIR / './models/xgb_model3.joblib')
    rbf_model = joblib.load(MODEL_DIR / './models/rbf_model.joblib')
    ridge_model = joblib.load(MODEL_DIR / './models/ridge_model.joblib')

    model = StackingRegressor(
        estimators=[
            ('ridge', ridge_model),
            ('xgb', xgb_model),
            ('xgb2', xgb_model2),
            ('xgb3', xgb_model3),
            ('rbf', rbf_model)
        ],
        cv=5,
        verbose=1,
    )

    model.fit(X, y)

    joblib.dump(model, MODEL_DIR / 'models/stacking_model.joblib')


if __name__ == '__main__':
    main()
