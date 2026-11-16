from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
FEATURES = ['Shipping Mode', 'Days for shipment (scheduled)', 'Order Region', 'Order Item Quantity', 'Order Item Total', 'Category Name']

st.set_page_config(page_title='CargoRisk AI', page_icon='🚚', layout='wide')
st.markdown('''<style>
.block-container {padding-top: 2rem; max-width: 1200px}
.hero {background:linear-gradient(110deg,#142c4e,#2265a4);padding:26px 30px;border-radius:18px;color:white;margin-bottom:18px}
.hero h1 {color:white;margin:0;font-size:2.25rem}.hero p{color:#dceaff;margin:8px 0 0}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🚚 CargoRisk AI</h1><p>Shipment delay risk • Order prioritization • Actionable recommendations</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return (joblib.load(ROOT / 'cargorisk_model.pkl'),
            joblib.load(ROOT / 'model_features.pkl'),
            json.loads((ROOT / 'model_metadata.json').read_text()))

try:
    model, model_columns, meta = load_assets()
except Exception as exc:
    st.error(f'Model files could not be loaded: {exc}')
    st.stop()

def recommend(risk, mode, days):
    if risk == 'Low Risk':
        return 'Monitor shipment normally.'
    if risk == 'Medium Risk':
        return 'Contact carrier and confirm shipping schedule.'
    if days <= 1:
        return 'Urgent: Verify same-day dispatch and escalate delays.'
    if mode == 'Standard Class':
        return 'Review faster shipping options and contact carrier.'
    if mode == 'First Class':
        return 'Escalate to carrier and verify dispatch priority.'
    return 'Urgent: Contact carrier and monitor shipment closely.'

def predict(data):
    encoded = pd.get_dummies(data[FEATURES], dtype=int)
    encoded = encoded.reindex(columns=model_columns, fill_value=0)
    probs = model.predict_proba(encoded)[:, 1]
    out = data.copy()
    out['Delay_Probability'] = probs
    out['Priority_Score'] = (probs * 100).round(2)
    out['Risk_Level'] = pd.cut(probs, bins=[-0.001, 0.4, 0.7, 1.001], labels=['Low Risk', 'Medium Risk', 'High Risk'], right=False).astype(str)
    out['Recommendation'] = [recommend(r, m, d) for r, m, d in zip(out['Risk_Level'], out['Shipping Mode'], out['Days for shipment (scheduled)'])]
    return out

with st.sidebar:
    st.header('About the model')
    st.write('Random Forest classifier')
    st.metric('Held-out ROC-AUC', f"{meta['roc_auc']:.3f}")
    st.metric('Held-out accuracy', f"{meta['accuracy_at_0_4']:.1%}")
    st.caption(f"Trained on {meta['training_rows']:,} order-item rows; evaluated on {meta['test_rows']:,} rows with orders separated between train and test.")
    st.info('Risk bands are provisional. Predictions estimate the DataCo late-shipping label, not guaranteed delivery dates.')

tab1, tab2 = st.tabs(['🔎 Predict a shipment', '📊 Analyze a CSV'])
with tab1:
    st.subheader('New shipment details')
    with st.form('shipment_form'):
        a, b = st.columns(2)
        with a:
            mode = st.selectbox('Shipping mode', meta['options']['Shipping Mode'])
            region = st.selectbox('Order region', meta['options']['Order Region'])
            category = st.selectbox('Product category', meta['options']['Category Name'])
        with b:
            days = st.number_input('Scheduled shipping days', min_value=0, max_value=60, value=4, step=1)
            quantity = st.number_input('Order item quantity', min_value=1, max_value=100000, value=1, step=1)
            value = st.number_input('Order item total (dataset currency)', min_value=0.0, value=150.0, step=10.0)
        submitted = st.form_submit_button('Analyze shipment', type='primary', use_container_width=True)
    if submitted:
        sample = pd.DataFrame([{'Shipping Mode':mode,'Days for shipment (scheduled)':days,'Order Region':region,'Order Item Quantity':quantity,'Order Item Total':value,'Category Name':category}])
        result = predict(sample).iloc[0]
        c1,c2,c3 = st.columns(3)
        c1.metric('Model risk score', f"{result['Priority_Score']:.1f}/100")
        c2.metric('Risk level', result['Risk_Level'])
        c3.metric('Suggested priority', 'Review urgently' if result['Risk_Level']=='High Risk' else 'Review' if result['Risk_Level']=='Medium Risk' else 'Routine')
        st.progress(float(result['Delay_Probability']))
        st.subheader('Recommended action')
        st.info(result['Recommendation'])
        st.caption('The score is an estimated model probability, not a guarantee. Suggestions are rule-based and should be reviewed by a human.')

with tab2:
    st.subheader('Bulk shipment assessment')
    st.write('Upload a CSV containing these six fields: ' + ', '.join(FEATURES) + '. Optional: `Order Id`.')
    template = pd.DataFrame([{'Shipping Mode':'Standard Class','Days for shipment (scheduled)':4,'Order Region':'South Asia','Order Item Quantity':1,'Order Item Total':150.0,'Category Name':'Cleats','Order Id':12345}])
    st.download_button('Download CSV template', template.to_csv(index=False).encode(), 'cargorisk_template.csv', 'text/csv')
    upload = st.file_uploader('Choose a shipment CSV', type='csv')
    if upload:
        try:
            incoming = pd.read_csv(upload)
            missing = [c for c in FEATURES if c not in incoming.columns]
            if missing:
                st.error('Missing columns: ' + ', '.join(missing))
            else:
                valid = incoming.dropna(subset=FEATURES).copy()
                for col in ['Days for shipment (scheduled)','Order Item Quantity','Order Item Total']:
                    valid[col] = pd.to_numeric(valid[col], errors='coerce')
                valid = valid.dropna(subset=FEATURES)
                if valid.empty:
                    st.warning('No valid rows found.')
                else:
                    result = predict(valid)
                    if 'Order Id' in result.columns:
                        # Highest item risk determines the order-level risk; total value sums line items.
                        result = result.sort_values('Delay_Probability',ascending=False)
                        grouped = result.groupby('Order Id',as_index=False).agg({
                            'Shipping Mode':'first','Days for shipment (scheduled)':'first',
                            'Order Item Total':'sum','Delay_Probability':'max',
                            'Priority_Score':'max','Risk_Level':'first','Recommendation':'first'})
                        grouped = grouped.rename(columns={'Order Item Total':'Order_Value'})
                        result = grouped
                    result = result.sort_values('Delay_Probability', ascending=False).reset_index(drop=True)
                    result['Priority_Rank'] = result['Delay_Probability'].rank(method='dense',ascending=False).astype(int)
                    m1,m2,m3 = st.columns(3)
                    m1.metric('Orders / rows assessed',f'{len(result):,}')
                    m2.metric('High risk',f"{(result['Risk_Level']=='High Risk').sum():,}")
                    m3.metric('Average risk score',f"{result['Priority_Score'].mean():.1f}/100")
                    st.subheader('Risk distribution')
                    counts = result['Risk_Level'].value_counts().reindex(['Low Risk','Medium Risk','High Risk'],fill_value=0)
                    st.bar_chart(counts)
                    st.subheader('Prioritized shipment report')
                    st.dataframe(result, use_container_width=True, hide_index=True)
                    st.download_button('Download prioritized report', result.to_csv(index=False).encode(), 'cargorisk_priority_report.csv', 'text/csv',type='primary')
                    if len(valid)!=len(incoming):
                        st.warning(f'{len(incoming)-len(valid)} incomplete or invalid rows were skipped.')
        except Exception as exc:
            st.error(f'Could not analyze CSV: {exc}')

st.divider()
st.caption('CargoRisk AI | Portfolio demonstration using historical DataCo Smart Supply Chain data. Not a production shipping decision system.')
