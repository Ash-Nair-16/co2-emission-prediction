# CO₂ Emission Prediction System

A machine learning-based web application that predicts **vehicle CO₂ emissions** based on engine size, fuel type, and other vehicle specifications.  
This project demonstrates the complete lifecycle of a predictive ML model — from **data preprocessing** and **model training** to **deployment on Render** with interactive visualization.

---

#Overview

With increasing concern over environmental impact, accurate prediction of CO₂ emissions is essential for vehicle manufacturers and environmental researchers.  
This system uses a **Random Forest Regressor** model to estimate CO₂ emissions from automotive datasets and provides insights into key contributing factors.

---

# Tech Stack

- **Language:** Python  
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn  
- **Model:** Random Forest Regressor  
- **Deployment:** [Render](https://render.com/)  
- **Visualization:** PowerBI Dashboard (for data insights)

---

# Features

- Data preprocessing using **OneHotEncoder** and feature scaling  
- Trained and evaluated **Random Forest Regressor** for high prediction accuracy  
- Visualized relationships between engine parameters and emissions using PowerBI  
- Web application interface deployed on **Render** for real-time predictions  
- Low-latency API endpoint for user inputs and model inference

---

#  Dataset

- Dataset Source: Publicly available automotive CO₂ emission dataset  
- Key Attributes:
  - Engine Size  
  - Cylinders  
  - Fuel Type  
  - Fuel Consumption (City / Highway / Combined)  
  - CO₂ Emissions (Target Variable)

---

# Deployment

The application is live on **Render** and can be accessed via:  
👉 [Live Demo Link](https://dashboard.render.com/web/srv-d3574ed6ubrc73cusv00/events)  

The model is integrated into a Flask-based backend, which handles user inputs and prediction requests.

---

##  Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ash-Nair-16/co2-emission-prediction.git
   cd co2-emission-prediction
