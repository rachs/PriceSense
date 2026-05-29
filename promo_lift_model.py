"""
promo_lift_model.py
-------------------
Standalone pipeline for per-UPC log-linear promo lift modelling.

Functions
---------
load_data               – read raw product and transaction CSVs
preprocess_data         – merge, clean, feature-engineer, train/test split
fit_promo_lift_models   – fit one OLS model per UPC with promo×month interactions
get_performance_metrics – evaluate fitted models against a held-out set
save_models             – persist model_store and promo_lift_df to disk
load_models             – restore model_store and promo_lift_df from disk
"""

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_data(
    prod_path: str = r"data\dunnhumby_prod.csv",
    trans_path: str = r"data\dunnhumby_trans.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the raw product and transaction CSV files.

    Parameters
    ----------
    prod_path  : path to the product metadata CSV
    trans_path : path to the transaction CSV

    Returns
    -------
    df_prod  : product metadata DataFrame
    df_trans : transaction DataFrame
    """
    df_prod  = pd.read_csv(prod_path)
    df_trans = pd.read_csv(trans_path)
    return df_prod, df_trans


# ---------------------------------------------------------------------------
# 2. Preprocess
# ---------------------------------------------------------------------------

def _derive_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add a PRICE column from whichever revenue column is present."""
    if "PRICE" in df.columns:
        return df

    for rev_col in ("DOLLARS", "SALES", "SPEND"):
        if rev_col in df.columns:
            df = df.copy()
            df["PRICE"] = np.where(df["UNITS"] > 0, df[rev_col] / df["UNITS"], np.nan)
            return df

    raise KeyError(
        "No usable price column found. Expected one of: PRICE, DOLLARS, SALES, SPEND."
    )


def preprocess_data(
    df_prod: pd.DataFrame,
    df_trans: pd.DataFrame,
    test_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge, clean, and split into per-UPC chronological train / test sets.

    Steps
    -----
    1. Left-join transactions to product metadata on UPC.
    2. Parse WEEK_END_DATE to datetime.
    3. Drop rows where WEEK_END_DATE is null or UNITS <= 0 (log(0) undefined).
    4. Derive PRICE when the column is absent (DOLLARS / UNITS, etc.).
    5. Add LOG_UNITS and LOG_PRICE.
    6. For each UPC, sort by date and put the last ``test_frac`` rows in the
       test set (at least 1 row) and the rest in train.

    Parameters
    ----------
    df_prod   : product metadata DataFrame (from load_data)
    df_trans  : transaction DataFrame (from load_data)
    test_frac : fraction of each UPC's rows reserved for the test set

    Returns
    -------
    df_train : training DataFrame
    df_test  : test DataFrame
    """
    df = pd.merge(df_trans, df_prod, on="UPC", how="left")
    df["WEEK_END_DATE"] = pd.to_datetime(df["WEEK_END_DATE"])

    # Drop unusable rows
    df = df[df["WEEK_END_DATE"].notna()]
    df = df[df["UNITS"].notna() & (df["UNITS"] > 0)]

    # Ensure PRICE exists
    df = _derive_price(df)

    # Log-transforms
    df["LOG_UNITS"] = np.where(df["UNITS"] > 0,  np.log(df["UNITS"]),  np.nan)
    df["LOG_PRICE"] = np.where(df["PRICE"] > 0,  np.log(df["PRICE"]),  np.nan)

    df = df.sort_values(["UPC", "WEEK_END_DATE"]).reset_index(drop=True)

    # Per-UPC chronological split
    train_parts, test_parts = [], []
    for _, group in df.groupby("UPC", sort=False):
        n        = len(group)
        n_test   = max(1, int(np.ceil(n * test_frac)))
        n_train  = n - n_test
        train_parts.append(group.iloc[:n_train])
        test_parts.append(group.iloc[n_train:])

    df_train = pd.concat(train_parts).reset_index(drop=True)
    df_test  = pd.concat(test_parts).reset_index(drop=True)

    print(
        f"Total rows : {len(df):,}\n"
        f"Train rows : {len(df_train):,}  ({100 * len(df_train) / len(df):.1f}%)\n"
        f"Test  rows : {len(df_test):,}   ({100 * len(df_test)  / len(df):.1f}%)"
    )

    return df_train, df_test


# ---------------------------------------------------------------------------
# 3. Fit per-UPC promo lift models
# ---------------------------------------------------------------------------

def fit_promo_lift_models(
    df: pd.DataFrame,
    min_obs: int = 20,
) -> tuple[dict, pd.DataFrame]:
    """
    Fit one OLS model per UPC with promo×month interactions:

        log(UNITS) = α + β·log(PRICE) + C(MONTH)
                       + FEATURE:C(MONTH) + DISPLAY:C(MONTH) + TPR_ONLY:C(MONTH) + ε

    Promo lift therefore varies by calendar month; each promo type yields 12
    separate lift estimates.

    Parameters
    ----------
    df      : training DataFrame (output of preprocess_data)
    min_obs : minimum number of valid rows required to fit a model for a UPC

    Returns
    -------
    model_store   : dict mapping UPC → fitted statsmodels OLS result
    promo_lift_df : DataFrame with one row per UPC; lift columns are named
                    ``feature_lift_pct_m1`` … ``tpr_lift_pct_m12``
    """
    required_cols = {"UPC", "UNITS", "PRICE", "FEATURE", "DISPLAY", "TPR_ONLY", "WEEK_END_DATE", "BASE_PRICE"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    model_store = {}
    rows        = []
    formula     = (
        "LOG_UNITS ~ LOG_PRICE + C(MONTH) "
        "+ FEATURE:C(MONTH) + DISPLAY:C(MONTH) + TPR_ONLY:C(MONTH)"
    )

    for upc, group in df.groupby("UPC"):
        sub = (
            group[["UNITS", "PRICE", "FEATURE", "DISPLAY", "TPR_ONLY", "WEEK_END_DATE"]]
            .dropna()
            .copy()
        )
        sub = sub[(sub["UNITS"] > 0) & (sub["PRICE"] > 0)]

        if len(sub) < min_obs:
            continue

        sub["LOG_UNITS"] = np.log(sub["UNITS"])
        sub["LOG_PRICE"] = np.log(sub["PRICE"])
        sub["MONTH"]     = pd.to_datetime(sub["WEEK_END_DATE"]).dt.month

        model      = smf.ols(formula, data=sub).fit()
        model_store[upc] = model

        row = {
            "UPC":              upc,
            "price_elasticity": model.params.get("LOG_PRICE", np.nan),
            "r_squared":        model.rsquared,
            "n_obs":            int(len(sub)),
        }

        # Per-month lift for each promo type
        for promo, col_prefix in [
            ("FEATURE", "feature"),
            ("DISPLAY", "display"),
            ("TPR_ONLY", "tpr"),
        ]:
            for m in range(1, 13):
                # Month 1 is the reference level; its main effect is the base coefficient
                param_name = promo if m == 1 else f"{promo}:C(MONTH)[T.{m}]"
                beta = model.params.get(param_name, np.nan)
                row[f"{col_prefix}_lift_pct_m{m}"] = (
                    (np.exp(beta) - 1) * 100 if pd.notna(beta) else np.nan
                )

        rows.append(row)

    promo_lift_df = pd.DataFrame(rows).sort_values("UPC").reset_index(drop=True)
    print(f"Products with sufficient promo data: {len(promo_lift_df)}")
    return model_store, promo_lift_df


# ---------------------------------------------------------------------------
# 4. Performance metrics
# ---------------------------------------------------------------------------

def get_performance_metrics(
    df_test: pd.DataFrame,
    model_store: dict,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate the per-UPC promo lift models against a held-out test set.

    For each UPC in df_test that has a fitted model the function:
    - predicts LOG_UNITS and back-transforms to unit scale
    - computes MAE, RMSE, R² on log scale and MAE, RMSE, MAPE on unit scale

    Parameters
    ----------
    df_test      : test DataFrame (output of preprocess_data)
    model_store  : dict mapping UPC → fitted OLS result (from fit_promo_lift_models)

    Returns
    -------
    agg_metrics  : dict of aggregate metrics across all evaluated rows
    per_upc_df   : DataFrame with per-UPC metrics
    """
    required_cols = {"UPC", "UNITS", "PRICE", "FEATURE", "DISPLAY", "TPR_ONLY", "WEEK_END_DATE"}
    missing = required_cols - set(df_test.columns)
    if missing:
        raise KeyError(f"df_test is missing columns: {sorted(missing)}")

    all_y_true_log,   all_y_pred_log   = [], []
    all_y_true_units, all_y_pred_units = [], []
    per_upc_rows = []

    for upc, group in df_test.groupby("UPC"):
        if upc not in model_store:
            continue

        sub = (
            group[["UNITS", "PRICE", "FEATURE", "DISPLAY", "TPR_ONLY", "WEEK_END_DATE"]]
            .dropna()
            .copy()
        )
        sub = sub[(sub["UNITS"] > 0) & (sub["PRICE"] > 0)]
        if sub.empty:
            continue

        sub["LOG_UNITS"] = np.log(sub["UNITS"])
        sub["LOG_PRICE"] = np.log(sub["PRICE"])
        sub["MONTH"]     = pd.to_datetime(sub["WEEK_END_DATE"]).dt.month

        y_pred_log   = model_store[upc].predict(sub).values
        y_true_log   = sub["LOG_UNITS"].values
        y_pred_units = np.exp(y_pred_log)
        y_true_units = sub["UNITS"].values

        mae_log    = mean_absolute_error(y_true_log,   y_pred_log)
        rmse_log   = root_mean_squared_error(y_true_log,   y_pred_log)
        r2_log     = r2_score(y_true_log,   y_pred_log)
        mae_units  = mean_absolute_error(y_true_units, y_pred_units)
        rmse_units = root_mean_squared_error(y_true_units, y_pred_units)
        mape       = np.mean(np.abs((y_true_units - y_pred_units) / y_true_units)) * 100

        per_upc_rows.append({
            "UPC":        upc,
            "n_test_obs": len(sub),
            "MAE_log":    round(mae_log,    4),
            "RMSE_log":   round(rmse_log,   4),
            "R2_log":     round(r2_log,     4),
            "MAE_units":  round(mae_units,  2),
            "RMSE_units": round(rmse_units, 2),
            "MAPE_pct":   round(mape,       2),
        })

        all_y_true_log.extend(y_true_log)
        all_y_pred_log.extend(y_pred_log)
        all_y_true_units.extend(y_true_units)
        all_y_pred_units.extend(y_pred_units)

    if not all_y_true_log:
        print("No overlapping UPCs found between df_test and model_store.")
        return {}, pd.DataFrame()

    agg_metrics = {
        "n_upcs_evaluated": len(per_upc_rows),
        "n_test_rows":      len(all_y_true_log),
        "MAE_log":          round(mean_absolute_error(all_y_true_log, all_y_pred_log),   4),
        "RMSE_log":         round(root_mean_squared_error(all_y_true_log, all_y_pred_log), 4),
        "R2_log":           round(r2_score(all_y_true_log, all_y_pred_log),              4),
        "MAE_units":        round(mean_absolute_error(all_y_true_units, all_y_pred_units), 2),
        "RMSE_units":       round(root_mean_squared_error(all_y_true_units, all_y_pred_units), 2),
        "MAPE_pct":         round(
            np.mean(
                np.abs(
                    (np.array(all_y_true_units) - np.array(all_y_pred_units))
                    / np.array(all_y_true_units)
                )
            ) * 100,
            2,
        ),
    }

    per_upc_df = pd.DataFrame(per_upc_rows).sort_values("UPC").reset_index(drop=True)
    return agg_metrics, per_upc_df


# ---------------------------------------------------------------------------
# 5. Save / load model store
# ---------------------------------------------------------------------------

def save_models(
    model_store: dict,
    promo_lift_df: pd.DataFrame,
    path: str = "promo_models.joblib",
) -> None:
    """
    Persist the model store and the promo lift summary DataFrame to a single
    file using joblib.

    Parameters
    ----------
    model_store   : dict mapping UPC to fitted OLS result
    promo_lift_df : DataFrame produced by fit_promo_lift_models
    path          : destination file path (default: ``promo_models.joblib``)
    """
    payload = {"model_store": model_store, "promo_lift_df": promo_lift_df}
    joblib.dump(payload, path)
    print(f"Saved {len(model_store)} models to '{path}'")


def load_models(path: str = "promo_models.joblib") -> tuple[dict, pd.DataFrame]:
    """
    Restore model store and promo lift DataFrame from a file saved by
    ``save_models``.

    Parameters
    ----------
    path : path to the joblib file created by save_models

    Returns
    -------
    model_store   : dict mapping UPC to fitted OLS result
    promo_lift_df : DataFrame with per-UPC lift estimates
    """
    payload = joblib.load(path)
    model_store   = payload["model_store"]
    promo_lift_df = payload["promo_lift_df"]
    print(f"Loaded {len(model_store)} models from '{path}'")
    return model_store, promo_lift_df


# ---------------------------------------------------------------------------
# Entry point – runs the full pipeline when executed as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_prod, df_trans = load_data()

    df_train, df_test = preprocess_data(df_prod, df_trans)

    model_store, promo_lift_df = fit_promo_lift_models(df_train, min_obs=20)
    print(promo_lift_df.head())

    save_models(model_store, promo_lift_df)

    agg_metrics, per_upc_metrics = get_performance_metrics(df_test, model_store)
    print("\n=== Aggregate performance on df_test ===")
    for k, v in agg_metrics.items():
        print(f"  {k}: {v}")
    print(per_upc_metrics.head(10))
