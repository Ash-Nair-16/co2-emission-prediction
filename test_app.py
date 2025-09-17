import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# =========================
# Helper: Matplotlib Gauge for PDF
# =========================
def generate_static_gauge(value, avg, safe_limit=150, moderate_limit=250, max_val=400):
    fig, ax = plt.subplots(figsize=(4, 2), subplot_kw={'projection': 'polar'})
    theta = np.linspace(0, np.pi, 100)

    # Zones
    ax.bar(theta[:33], np.ones(33), width=np.pi/100, color="green", alpha=0.6)
    ax.bar(theta[33:66], np.ones(33), width=np.pi/100, color="orange", alpha=0.6)
    ax.bar(theta[66:], np.ones(34), width=np.pi/100, color="red", alpha=0.6)

    # Needle for prediction
    angle_pred = np.pi * value / max_val
    ax.arrow(angle_pred, 0, 0, 0.9, width=0.02, head_width=0.06, head_length=0.1,
             fc="black", ec="black")

    # Needle for class average
    angle_avg = np.pi * avg / max_val
    ax.arrow(angle_avg, 0, 0, 0.8, width=0.01, head_width=0.05, head_length=0.08,
             fc="blue", ec="blue")

    # Clean chart
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# =========================
# Load model & dataset
# =========================
model = joblib.load("best_co2_model.pkl")
df = pd.read_csv("co2 Emissions.csv")

st.set_page_config(page_title="🌍 CO₂ Emission Predictor", layout="wide")

# =========================
# Header
# =========================
st.title("🌍 CO₂ Emission Predictor")
st.write("Enter vehicle details below to estimate *CO₂ emissions (g/km)*, compare with averages, and download a detailed report.")

# Sidebar inputs
st.sidebar.header("🚗 Vehicle Inputs")

make = st.sidebar.text_input("Car Make", "Toyota")
model_name = st.sidebar.text_input("Model", "Corolla")
vehicle_class = st.sidebar.selectbox("Vehicle Class", df["Vehicle Class"].unique())
engine_size = st.sidebar.number_input("Engine Size (L)", min_value=1.0, max_value=8.0, step=0.1)
cylinders = st.sidebar.number_input("Cylinders", min_value=3, max_value=16, step=1)
transmission = st.sidebar.selectbox("Transmission", df["Transmission"].unique())
fuel_type = st.sidebar.selectbox("Fuel Type", df["Fuel Type"].unique())
fuel_city = st.sidebar.number_input("Fuel Consumption City (L/100 km)", min_value=2.0, max_value=30.0, step=0.1)
fuel_hwy = st.sidebar.number_input("Fuel Consumption Hwy (L/100 km)", min_value=2.0, max_value=25.0, step=0.1)
fuel_comb = st.sidebar.number_input("Fuel Consumption Comb (L/100 km)", min_value=2.0, max_value=25.0, step=0.1)
fuel_mpg = st.sidebar.number_input("Fuel Consumption Comb (mpg)", min_value=5, max_value=80, step=1)

# =========================
# Prediction
# =========================
if st.sidebar.button("🔮 Predict CO₂ Emission"):
    input_df = pd.DataFrame([{
        "Make": make,
        "Model": model_name,
        "Vehicle Class": vehicle_class,
        "Engine Size(L)": engine_size,
        "Cylinders": cylinders,
        "Transmission": transmission,
        "Fuel Type": fuel_type,
        "Fuel Consumption City (L/100 km)": fuel_city,
        "Fuel Consumption Hwy (L/100 km)": fuel_hwy,
        "Fuel Consumption Comb (L/100 km)": fuel_comb,
        "Fuel Consumption Comb (mpg)": fuel_mpg
    }])

    prediction = model.predict(input_df)[0]

    # Compute class average
    avg_class_emission = df[df["Vehicle Class"] == vehicle_class]["CO2 Emissions(g/km)"].mean()

    # Classify emissions
    if prediction < 150:
        status = "✅ Safe"
        color = "green"
        tips = ["Keep driving efficiently!", "Maintain your vehicle regularly."]
    elif prediction <= 250:
        status = "⚠ Moderate"
        color = "orange"
        tips = [
            "Avoid aggressive driving.",
            "Use cruise control when possible.",
            "Reduce extra weight in the car."
        ]
    else:
        status = "❌ High"
        color = "red"
        tips = [
            "Consider hybrid or electric vehicles.",
            "Carpool or use public transport.",
            "Limit AC usage when not needed."
        ]

    # =========================
    # Show Results
    # =========================
    st.markdown(
        f"## Estimated CO₂ Emission: <span style='color:{color}'>{prediction:.2f} g/km</span>",
        unsafe_allow_html=True
    )
    st.markdown(f"### Emission Level: *{status}*")

    # =========================
    # Plotly Gauge Chart
    # =========================
    st.subheader("📊 CO₂ Emission Gauge")

    safe_limit = 150
    moderate_limit = 250
    max_val = 400

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        title={'text': "CO₂ Emissions (g/km)"},
        delta={
            'reference': avg_class_emission,
            'relative': False,
            'increasing': {'color': "red"},
            'decreasing': {'color': "green"}
        },
        gauge={
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "black"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, safe_limit], 'color': "green"},
                {'range': [safe_limit, moderate_limit], 'color': "orange"},
                {'range': [moderate_limit, max_val], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "blue", 'width': 4},
                'thickness': 0.75,
                'value': avg_class_emission
            }
        }
    ))

    fig.update_layout(
        height=380,
        margin=dict(l=40, r=40, t=80, b=40),
        font=dict(size=16)
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # Tips
    # =========================
    st.subheader("💡 Eco-Friendly Suggestions")
    for t in tips:
        st.write(f"- {t}")

    # =========================
    # Download Report (PDF)
    # =========================
    st.subheader("📥 Download Your Report")

    def generate_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph(" CO2 Emission Report", styles["Title"]))
        elements.append(Spacer(1, 12))

        # Vehicle details
        elements.append(Paragraph(" Vehicle Details", styles["Heading2"]))
        details = f"""
        Make: {make}<br/>
        Model: {model_name}<br/>
        Vehicle Class: {vehicle_class}<br/>
        Engine Size: {engine_size} L<br/>
        Cylinders: {cylinders}<br/>
        Transmission: {transmission}<br/>
        Fuel Type: {fuel_type}<br/>
        """
        elements.append(Paragraph(details, styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Prediction
        elements.append(Paragraph(" Prediction", styles["Heading2"]))
        elements.append(Paragraph(f"Predicted CO2 Emission: <b>{prediction:.2f} g/km</b>", styles["Normal"]))
        elements.append(Paragraph(f"Emission Status: <b>{status}</b>", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Gauge Chart (Matplotlib static version for PDF)
        elements.append(Paragraph(" Emission Gauge", styles["Heading2"]))
        static_chart = generate_static_gauge(prediction, avg_class_emission)
        chart_img = Image(static_chart, width=400, height=200)
        elements.append(chart_img)
        elements.append(Spacer(1, 12))

        # Tips
        elements.append(Paragraph(" Eco-Friendly Suggestions:", styles["Heading2"]))
        for t in tips:
            elements.append(Paragraph(f"- {t}", styles["Normal"]))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf_buffer = generate_pdf()

    st.download_button(
        label="⬇ Download Report (PDF)",
        data=pdf_buffer,
        file_name="co2_emission_report.pdf",
        mime="application/pdf"
    )