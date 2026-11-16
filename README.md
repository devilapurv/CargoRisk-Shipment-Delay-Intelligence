# CargoRisk AI — Shipment Delay Intelligence

A Streamlit ML portfolio app to estimate late-shipping risk, rank shipments, and suggest follow-up actions.

## Run locally

Open a terminal inside this extracted folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501 (usually automatically).

## Features

- Predict shipment late-shipping risk from six inputs
- Show Low / Medium / High risk and a 0–100 priority score
- Generate rule-based recommendations
- Upload a CSV to assess many shipment rows
- Consolidate repeated `Order Id` values and download an order-priority report

Download the sample CSV template directly from the app.

## Model and limitations

The ZIP includes a pretrained Random Forest and feature schema. This **is not the exact Colab model**: it was retrained on the available 50,019-row DataCo CSV (50,018 rows after dropping missing fields), with an order-grouped 80/20 holdout. See `model_metadata.json` for measured performance. The app does not need the training CSV to run.

The outcome is the dataset's `Late_delivery_risk` label, which is not necessarily the same as a real last-mile delivery failure. Risk scores and bands are provisional and should not be used for real shipping decisions without independent validation. Scores may not be well calibrated for new populations. The recommendations are simple rules, not demonstrated interventions.

When multiple line items share an Order ID, the bulk dashboard uses the highest line-item predicted risk and sums the order-item totals. This is a heuristic, not a separately trained order-level model.

If you prefer your **exact Colab model**, replace `cargorisk_model.pkl` and `model_features.pkl` with the files saved in Colab. Their scikit-learn versions should be compatible.
