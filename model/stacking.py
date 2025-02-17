from pathlib import Path
from os import path
import numpy as np
import joblib
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_validate

from .core import MODEL_DIR, load_Xy


def main():
    X, y = load_Xy()
    xgb_model = joblib.load(MODEL_DIR / './models/xgb_model.joblib')
    #xgb_model2 = joblib.load(MODEL_DIR / './models/xgb_model2.joblib')
    #xgb_model3 = joblib.load(MODEL_DIR / './models/xgb_model3.joblib')
    svr_model = joblib.load(MODEL_DIR / './models/linear_svr_model.joblib')
    rbf_model = joblib.load(MODEL_DIR / './models/rbf_model.joblib')
    ridge_model = joblib.load(MODEL_DIR / './models/ridge_model.joblib')
    elasticnet_model = joblib.load(MODEL_DIR / './models/elasticnet_model.joblib')

    model = StackingRegressor(
        estimators=[
            ('ridge', ridge_model),
            ('elasticnet', elasticnet_model),
            ('xgb', xgb_model),
            #('xgb2', xgb_model2),
            #('xgb3', xgb_model3),
            #('svr', svr_model),
            ('rbf', rbf_model),
        ],
        cv=5,
        verbose=1,
    )
    cv = cross_validate(model, X, y, scoring='neg_mean_squared_error')
    joblib.dump(cv, MODEL_DIR / 'model_results/stacking_cv.joblib')
    print(cv)
    print(np.mean(cv['test_score']))

    model.fit(X, y)

    joblib.dump(model, MODEL_DIR / 'models/stacking_model.joblib')


if __name__ == '__main__':
    main()
