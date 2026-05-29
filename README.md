# PriceSense AI

PriceSense AI predicts promotional unit lift for a product UPC at a given price, promo type, and calendar month.

## What’s included

- `app.py` — Streamlit UI for interactive lift prediction
- `promo_lift_model.py` — data loading, preprocessing, model training, and evaluation
- `predict_promo_lift.py` — helper for single and batch lift predictions
- `data\` — bundled Dunnhumby CSV files used by the model
- `promo_models.joblib` — saved model store generated after training

## Requirements

- Python 3.10+
- `pandas`
- `numpy`
- `statsmodels`
- `scikit-learn`
- `joblib`
- `streamlit`

## Quick start

1. Install the dependencies:

   ```bash
   pip install pandas numpy statsmodels scikit-learn joblib streamlit
   ```

2. Train and save the models:

   ```bash
   python promo_lift_model.py
   ```

3. Launch the app:

   ```bash
   streamlit run app.py
   ```

## How it works

The training script merges product and transaction data, derives price features, fits one OLS model per UPC, and saves the resulting model store to `promo_models.joblib`. The Streamlit app loads that file and lets you compare baseline units against a promo scenario.

## Input data

By default, the scripts read:

- `data\dunnhumby_prod.csv`
- `data\dunnhumby_trans.csv`

If you replace the data, keep the same column structure expected by the scripts.

## Notes

- The app shows a predicted baseline, promo units, unit lift, and lift percentage.
- Promo types supported by the predictor are `FEATURE`, `DISPLAY`, and `TPR_ONLY`.
