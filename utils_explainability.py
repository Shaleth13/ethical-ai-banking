from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline


def get_feature_contributions(model: Pipeline, input_df: pd.DataFrame) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    X_trans = preprocessor.transform(input_df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    X_row = X_trans[0]

    coefs = clf.coef_[0]
    contribs = coefs * X_row
    feature_names = preprocessor.get_feature_names_out()

    contrib_df = pd.DataFrame(
        {
            "feature_raw": feature_names,
            "contribution": contribs,
        }
    )
    contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
    contrib_df["feature"] = contrib_df["feature_raw"].str.replace(r"^num__|^cat__", "", regex=True)
    contrib_df["feature"] = contrib_df["feature"].str.replace("_", " ")

    contrib_df = contrib_df.sort_values("abs_contribution", ascending=False)
    return contrib_df[["feature", "contribution"]]


def summarize_input_for_log(input_dict: Dict) -> str:
    parts = [f"{k}={v}" for k, v in input_dict.items()]
    return "; ".join(parts)
