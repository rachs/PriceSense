"""
predict_promo_lift.py
---------------------
Given a trained per-UPC OLS model (from promo_lift_model.py), predict the
projected unit lift that a promotion would deliver at a specific price and
calendar month compared to the no-promo baseline at the same price.

Usage
-----
    from promo_lift_model import load_models
    from predict_promo_lift import predict_promo_lift

    model_store, _ = load_models("promo_models.joblib")
    result = predict_promo_lift(
        model=model_store[upc]["model"],
        upc=upc,
        new_price=2.99,
        promo_type="FEATURE",
        month=6,
    )
    print(result)

Or run as a script for a quick demo:
    python predict_promo_lift.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


VALID_PROMO_TYPES = {"FEATURE", "DISPLAY", "TPR_ONLY"}


def predict_promo_lift(
    model,
    upc,
    new_price: float,
    promo_type: str,
    month: int,
) -> dict | None:
    """
    Predict the unit lift that a promotion would deliver at a given price and
    calendar month, compared to the no-promo baseline at the same price.

    The function builds two one-row prediction DataFrames from the fitted OLS
    model:
      - **baseline row** – all promo flags set to 0 at ``new_price`` / ``month``
      - **promo row**    – the requested ``promo_type`` flag set to 1

    Both rows are passed through ``model.predict()`` and the results are
    back-transformed from log scale to unit scale.

    Parameters
    ----------
    model      : fitted statsmodels OLS result for the UPC
                 (e.g. ``model_store[upc]`` from ``fit_promo_lift_models``)
    upc        : product UPC identifier (used only for labelling the output)
    new_price  : proposed selling price in dollars (must be > 0)
    promo_type : one of ``'FEATURE'``, ``'DISPLAY'``, ``'TPR_ONLY'``
    month      : calendar month integer (1 = Jan … 12 = Dec)

    Returns
    -------
    dict with keys:
        upc            – product identifier
        new_price      – price used for prediction
        promo_type     – promotion type
        month          – calendar month
        baseline_units – predicted units with no promotion
        promo_units    – predicted units with promotion active
        unit_lift      – promo_units - baseline_units
        lift_pct       – percentage lift over baseline

    Returns ``None`` and prints a warning if ``new_price`` <= 0.
    """
    if promo_type not in VALID_PROMO_TYPES:
        raise ValueError(f"promo_type must be one of: {sorted(VALID_PROMO_TYPES)}")

    if not isinstance(month, int) or not (1 <= month <= 12):
        raise ValueError("month must be an integer between 1 and 12")

    if new_price <= 0:
        print(f"UPC {upc}: new_price must be > 0 (got {new_price}).")
        return None

    # Baseline: all promo flags off
    base_row = pd.DataFrame([{
        "LOG_PRICE": np.log(new_price),
        "FEATURE":   0.0,
        "DISPLAY":   0.0,
        "TPR_ONLY":  0.0,
        "MONTH":     month,
    }])

    # Promo row: flip the requested flag to 1
    promo_row = base_row.copy()
    promo_row[promo_type] = 1.0

    baseline_units = np.exp(model.predict(base_row).iloc[0])
    promo_units    = np.exp(model.predict(promo_row).iloc[0])

    unit_lift = promo_units - baseline_units
    lift_pct  = (promo_units / baseline_units - 1) * 100

    return {
        "upc":            upc,
        "new_price":      new_price,
        "promo_type":     promo_type,
        "month":          month,
        "baseline_units": round(baseline_units, 1),
        "promo_units":    round(promo_units,    1),
        "unit_lift":      round(unit_lift,      1),
        "lift_pct":       round(lift_pct,       2),
    }


def batch_predict_promo_lift(
    model_store: dict,
    requests: list[dict],
) -> pd.DataFrame:
    """
    Run ``predict_promo_lift`` for a list of (upc, new_price, promo_type, month)
    dicts and return results as a DataFrame.

    Each item in ``requests`` must contain the keys:
        ``upc``, ``new_price``, ``promo_type``, ``month``

    UPCs not present in ``model_store`` are skipped with a warning.

    Parameters
    ----------
    model_store : dict mapping UPC to fitted OLS result
    requests    : list of dicts, one per prediction to make

    Returns
    -------
    pd.DataFrame with one row per successful prediction
    """
    rows = []
    for req in requests:
        upc = req["upc"]
        if upc not in model_store:
            print(f"UPC {upc} not found in model_store – skipping.")
            continue
        result = predict_promo_lift(
            model      = model_store[upc]["model"],
            upc        = upc,
            new_price  = req["new_price"],
            promo_type = req["promo_type"],
            month      = req["month"],
        )
        if result is not None:
            rows.append(result)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Demo – runs when the script is executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys

    try:
        from promo_lift_model import load_models
    except ImportError:
        sys.exit("Could not import promo_lift_model. Run this script from the PriceSense directory.")

    model_path = "promo_models.joblib"
    if not os.path.exists(model_path):
        sys.exit(
            f"'{model_path}' not found. Run promo_lift_model.py first to train and save models."
        )

    model_store, _ = load_models(model_path)

    # sample_upc = next(iter(model_store))
    # Take the sample upc, the promo type, the new price and the month as arguments to the script, or 
    # default to the first UPC in the model_store
    import argparse
    parser = argparse.ArgumentParser(description="Predict promo lift for a sample UPC.")
    parser.add_argument("--upc", type=float, default=next(iter(model_store)), help="UPC to predict for")
    parser.add_argument("--new_price", type=float, default=2.99, help="New price for prediction")
    parser.add_argument("--promo_type", type=str, default="FEATURE", help="Promotion type (FEATURE, DISPLAY, TPR_ONLY)")
    parser.add_argument("--month", type=int, default=6, help="Calendar month (1-12)")
    args = parser.parse_args()      
    sample_upc = args.upc
    new_price = args.new_price
    promo_type = args.promo_type
    month = args.month

    result = predict_promo_lift(
        model      = model_store[sample_upc],
        upc        = sample_upc,
        new_price  = new_price,
        promo_type = promo_type,
        month      = month,
    )
    print("\n=== Single prediction ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # print("\n=== Batch predictions ===")
    # upcs = list(model_store.keys())[:3]
    # batch = batch_predict_promo_lift(
    #     model_store,
    #     [
    #         {"upc": upcs[0], "new_price": 2.49, "promo_type": "FEATURE",  "month": 3},
    #         {"upc": upcs[1], "new_price": 3.99, "promo_type": "DISPLAY",  "month": 7},
    #         {"upc": upcs[2], "new_price": 1.79, "promo_type": "TPR_ONLY", "month": 11},
    #     ],
    # )
    # print(batch.to_string(index=False))
