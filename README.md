# ⚡ Electricity Consumption Predictor

A beginner-friendly Machine Learning project that predicts electricity consumption using time, weather, and previous-consumption features.

## 📌 Project Overview

Electricity consumption changes according to factors such as time of day, day of week, temperature, humidity, and previous energy usage.

This project uses **Python, Pandas, NumPy, Matplotlib, Scikit-learn, and Streamlit** to build an Electricity Consumption Predictor.

### Project Flow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Data Visualization
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Random Forest Regression
   ↓
Model Evaluation
   ↓
Electricity Consumption Prediction
   ↓
Streamlit Web App
```

## 🧠 Machine Learning Model

**Random Forest Regression** is used to predict electricity consumption.

### Input Features

- Hour of day
- Day of week
- Weekend/weekday
- Temperature (°C)
- Humidity (%)
- Previous electricity consumption (kWh)

### Output

Predicted electricity consumption in **kWh**.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Scikit-learn | Machine Learning |
| Joblib | Model saving/loading |
| Streamlit | Web interface |

## 📂 Project Structure

```text
Electricity_Consumption_Predictor/
│
├── app.py
├── train_model.py
├── electricity_consumption.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Electricity_Consumption_Predictor.git
cd Electricity_Consumption_Predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This generates:

- `electricity_model.pkl`
- `actual_vs_predicted.png`
- `feature_importance.png`

These generated files are intentionally ignored by Git.

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Model Evaluation

The training script calculates:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R² Score** — Coefficient of Determination

The exact values depend on the dataset and training configuration.

## 📈 Visualizations

The project generates:

1. Actual vs Predicted Electricity Consumption
2. Feature Importance

## 📁 Dataset

The included dataset is an **educational demonstration dataset** generated for this project.

For a research/production version, replace it with real smart-meter or electricity-consumption data and retrain the model.

## 🔮 Future Scope

- Real-time smart-meter data
- IoT integration
- Appliance-level energy monitoring
- Cloud dashboard
- Mobile application
- Deep Learning/LSTM forecasting
- Solar generation prediction
- AI-based energy-saving recommendations

## 👨‍💻 Project Type

**Foundation / Academic Machine Learning Project**

Suitable for demonstrating:

- Python
- Python Libraries
- Data Visualization
- Machine Learning
- Basic AI concepts
- Project Development

## ⚠️ Disclaimer

This project is intended for educational purposes. Prediction performance on real electricity data may differ from the included demonstration dataset.
