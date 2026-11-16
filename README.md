# 🚚 CargoRisk AI — Shipment Delay Intelligence

**Predict shipping-delay risk, prioritize orders, and recommend follow-up actions with Machine Learning and Streamlit.**

CargoRisk AI is an end-to-end supply-chain analytics project that estimates the risk of **late shipping** from shipment and order details. It supports both individual shipment assessment and bulk CSV analysis, helping users identify orders that may need attention.

## 📸 Application screenshots

### 1. Individual shipment prediction
Choose shipping mode, scheduled shipping days, order region, category, quantity, and order value. The app displays a risk score, risk category, priority, and recommended action.

![Individual shipment prediction](screenshots/Dashboard1.png)

### 2. Bulk CSV assessment
Upload multiple shipments and explore the risk distribution, number of high-risk records, and average risk score.

![Bulk shipment assessment](screenshots/Dashboard2.png)

### 3. Prioritized shipment report
Review orders sorted by predicted risk, with shipping details and suggested follow-up actions. Download the report as CSV.

![Prioritized shipment report](screenshots/Dashboard3.png)

## ✨ Features

- **Shipment risk prediction:** Random Forest classification using six input fields.
- **Risk assessment:** Low, Medium, and High risk bands with a score from 0 to 100.
- **Actionable recommendations:** Rule-based suggestions informed by risk level and shipping mode.
- **Order prioritization:** Sort high-risk orders first for review.
- **Bulk analysis:** Upload a compatible CSV, view summary metrics and a risk-distribution chart, and export results.
- **Order consolidation:** When an uploaded CSV contains repeated Order IDs, combine line items into an order-level report.

## 📊 Dataset

**DataCo Smart Supply Chain Dataset** — available on [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis).

The project uses these fields:

| Column | Purpose |
|---|---|
| `Shipping Mode` | Shipment service class |
| `Days for shipment (scheduled)` | Planned shipping duration |
| `Order Region` | Destination region |
| `Order Item Quantity` | Quantity of items |
| `Order Item Total` | Value of the order line |
| `Category Name` | Product category |
| `Late_delivery_risk` | Training target: 0 = not late, 1 = late |

The trained model and app are included in the project; **the full dataset is not needed to run the app**. For bulk predictions, upload a CSV with the six input columns above (excluding the target). `Order Id` is optional and helps group repeated order lines. A sample template can be downloaded inside the app.

## 🧠 Machine learning approach

1. Select six shipment/order features and the `Late_delivery_risk` target.
2. Handle missing rows and one-hot encode categorical inputs.
3. Split training and testing data by **Order ID** to reduce leakage from repeated line items.
4. Train a **Random Forest Classifier**.
5. Convert model scores into risk categories and rule-based recommendations.
6. Display predictions and order-level summaries in Streamlit.

### Model evaluation

The **model bundled with this Streamlit ZIP** was retrained separately from the Google Colab experiments. Its `model_metadata.json` reports the following order-grouped holdout results:

| Metric | Bundled model |
|---|---:|
| Training rows | 39,944 |
| Test rows | 10,074 |
| Accuracy (0.40 threshold) | 68.5% |
| ROC-AUC | 0.764 |

*Note:* Earlier Colab experiments used a different data subset and produced different metrics. The values above describe the model distributed with this app, not those earlier experiments.

## 🛠️ Tech stack

**Python · Pandas · NumPy · Scikit-learn · Streamlit · Joblib**

## 🚀 Run locally

1. Download or clone the repository and open a terminal inside the project directory.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start Streamlit:

   ```bash
   python -m streamlit run app.py
   ```

4. Open **http://localhost:8501** if it does not open automatically.

> On Windows, `python -m pip` avoids problems caused by an outdated standalone `pip` launcher. The included `run_windows.bat` is another launch option.

## 📁 Project structure

```text
CargoRisk-AI/
├── app.py
├── requirements.txt
├── cargorisk_model.pkl
├── model_features.pkl
├── model_metadata.json
├── run_windows.bat
├── README.md
└── screenshots/
    ├── Dashboard1.png
    ├── Dashboard2.png
    └── Dashboard3.png
```

## ⚠️ Limitations

- The model predicts the dataset's **late-shipping risk label**, not a verified last-mile delivery delay.
- The 0–100 score is derived from a model probability estimate; it is **not a guarantee** or an independently validated operational probability.
- Risk thresholds and recommendations are heuristic and should be validated before real logistics decisions.
- The order-level score uses the **maximum line-item risk** for each order, not a separately trained order-level model.
- Predictions may not generalize to new carriers, regions, time periods, or shipping processes without additional testing.

## 📌 Future improvements

Time-based validation, more detailed probability calibration, richer shipment tracking data, and cost-aware order prioritization.
