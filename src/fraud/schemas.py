"""Request/response models. Pydantic does the input validation."""
from pydantic import BaseModel, ConfigDict, Field, create_model

from .serving import PCA_COLUMNS


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")      # unknown fields are an error, not silently ignored


_fields = {
    "Time": (float, Field(..., ge=0, le=1e9, allow_inf_nan=False,
                          description="Seconds since the first transaction in the dataset (drives hour-of-day).")),
    "Amount": (float, Field(..., ge=0, le=1e7, allow_inf_nan=False, description="Transaction amount.")),
    **{c: (float, Field(..., allow_inf_nan=False, description="Anonymised PCA component.")) for c in PCA_COLUMNS},
}
Transaction = create_model("Transaction", __base__=_Strict, **_fields)


class Prediction(BaseModel):
    fraud_probability: float = Field(description="Calibrated probability that the transaction is fraud.")
    raw_score: float = Field(description="Uncalibrated model score (the threshold applies to this).")
    is_fraud: bool = Field(description="True if raw_score >= threshold (cost-tuned decision rule).")
    threshold: float
    model_version: str


class FeatureContribution(BaseModel):
    feature: str
    contribution: float = Field(description="Exact LightGBM Tree SHAP contribution to the raw tree margin.")


class Explanation(BaseModel):
    raw_margin: float = Field(description="Uncalibrated LightGBM margin; contributions sum with base_value to this.")
    base_value: float
    contributions: list[FeatureContribution]
    model_version: str