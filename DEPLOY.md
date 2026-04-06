# 🚀 HeartGuard – Streamlit Deployment Guide

## Project Structure

```
heartguard_streamlit/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── heart.csv               ← Dataset
├── models/
│   ├── heartguard_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── importances.pkl
│   ├── importance_labels.pkl
│   └── model_metrics.json
├── static/img/             ← Pre-generated EDA plots
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── eda_distributions.png
│   ├── feature_importances.png
│   └── roc_curves.png
└── .streamlit/
    └── config.toml         ← Dark theme + server settings
```

---

## ✅ Option 1 – Run Locally

### Step 1: Install Python
Ensure you have **Python 3.9+** installed.  
Download from https://www.python.org/downloads/

### Step 2: Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the app
```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## ☁️ Option 2 – Deploy on Streamlit Community Cloud (FREE)

### Prerequisites
- Free account at https://share.streamlit.io
- Your code pushed to a **GitHub repository**

### Step-by-step

**1. Create a GitHub repository**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/heartguard.git
git push -u origin main
```

**2. Upload model files to GitHub**
The `models/` folder in this project is about **12.5 MB**, so it can be committed normally.
If future model files grow large, use Git LFS:
```bash
git lfs install
git lfs track "models/*.pkl"
git add .gitattributes
git commit -m "Track large files with LFS"
```

**3. Deploy on Streamlit Cloud**
1. Go to https://share.streamlit.io
2. Click **"New app"**
3. Select your GitHub repo → branch `main` → main file `app.py`
4. Click **"Deploy!"**

The app will be live at:  
`https://YOUR_USERNAME-heartguard-app-XXXX.streamlit.app`

---

## 🐳 Option 3 – Deploy with Docker

### Dockerfile
Create a file named `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]
```

### Build and run
```bash
docker build -t heartguard .
docker run -p 8501:8501 heartguard
```

Open **http://localhost:8501**

---

## ☁️ Option 4 – Deploy on Render (FREE tier)

1. Push your code to GitHub (see Option 2, Step 1)
2. Go to https://render.com → **New Web Service**
3. Connect your GitHub repo
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **Create Web Service**

---

## 🔄 Re-Training the Model (Optional)

This repository currently includes the trained model artifacts and generated plots, but it does **not** include a training script.
If you later add a `train_model.py` pipeline, update this section with the retraining steps and outputs.

---

## 🩺 App Features

| Page | Description |
|---|---|
| 🏠 Home | Project overview, key metrics, feature list |
| 🔬 Predict | Enter patient data → get real-time risk prediction |
| 📊 Analytics & EDA | Dataset explorer, plots, model comparison, feature importances |
| ℹ️ About | Project details, tech stack, disclaimer |

---

## ⚠️ Disclaimer

HeartGuard is a **research and educational tool**.  
It is NOT a certified medical device and must NOT replace professional medical consultation.

---

*Author: Kartikkumar Sangada (22012011044) | U.V. Patel College of Engineering | Ganpat University*
