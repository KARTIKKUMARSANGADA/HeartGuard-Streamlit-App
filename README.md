# HeartGuard

HeartGuard is a Streamlit app that predicts heart disease risk from 13 clinical inputs using a trained Random Forest model. The repository includes the web app, trained model artifacts, dataset, and pre-generated visualizations used in the dashboard.

## Features

- Real-time heart disease risk prediction
- Prediction confidence and risk-level summary
- Top contributing factors from the deployed model
- Dataset overview and EDA visuals
- Model comparison metrics and feature importances

## Project Structure

```text
heartguard_streamlit/
|-- app.py
|-- requirements.txt
|-- heart.csv
|-- DEPLOY.md
|-- models/
|   |-- heartguard_model.pkl
|   |-- scaler.pkl
|   |-- feature_names.pkl
|   |-- importances.pkl
|   |-- importance_labels.pkl
|   `-- model_metrics.json
|-- static/
|   `-- img/
`-- .streamlit/
    `-- config.toml
```

## Run Locally

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run app.py
```

## Deployment

Deployment steps for Streamlit Community Cloud and other options are in [DEPLOY.md](DEPLOY.md).

## Notes

- The trained model artifact in `models/heartguard_model.pkl` is about 12.5 MB, so it can be stored in a normal GitHub repository without Git LFS.
- `.streamlit/config.toml` is safe to commit. `.streamlit/secrets.toml` should stay local and is ignored.
- This project is an educational/research tool and not a certified medical device.
