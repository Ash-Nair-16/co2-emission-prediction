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

# Load trained model & dataset
model = joblib.load("best_co2_model.pkl")
df = pd.read_csv("co2 Emissions.csv")

st.set_page_config(page_title="🌍 CO₂ Emission Predictor", layout="wide")

# =========================
# Header
# =========================
st.title("🌍 CO₂ Emission Predictor")
st.write("Enter vehicle details below to estimate CO₂ emissions (g/km), compare with averages, and download a detailed report.")

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
    best_in_class = df[df["Vehicle Class"] == vehicle_class]["CO2 Emissions(g/km)"].min()

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
    # Gamification: Green Score
    # =========================
    max_emission = 400  # assume 400 g/km is worst case
    green_score = max(0, min(100, 100 - (prediction / max_emission) * 100))

    if green_score >= 70:
        badge = "🌱 Eco"
    elif green_score >= 40:
        badge = "⚡ Moderate"
    else:
        badge = "🔥 Polluter"

    st.subheader("🏆 Green Score")
    st.markdown(
        f"<h2 style='color:{color};'>Score: {green_score:.0f}/100 &nbsp; {badge}</h2>",
        unsafe_allow_html=True
    )
    st.progress(int(green_score))

    # =========================
    # Show Results
    # =========================
    st.markdown(
        f"## Estimated CO₂ Emission: <span style='color:{color}'>{prediction:.2f} g/km</span>",
        unsafe_allow_html=True
    )
    st.markdown(f"### Emission Level: {status}")

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

    # Save chart as image for PDF
    chart_buf = BytesIO()
    fig.write_image(chart_buf, format="png")
    chart_buf.seek(0)

    # =========================
    # Emission Comparison (Bar Chart)
    # =========================
    st.subheader("📊 Emission Comparison")

    comparison_data = {
        "Your Car": prediction,
        "Class Avg": avg_class_emission,
        "Best in Class": best_in_class
    }

    fig_bar, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(x=list(comparison_data.keys()), y=list(comparison_data.values()), palette="viridis", ax=ax)
    ax.set_ylabel("CO₂ Emissions (g/km)")
    ax.set_title("Emission Comparison")

    st.pyplot(fig_bar, use_container_width=True)

    # Save bar chart for PDF
    bar_buf = BytesIO()
    fig_bar.savefig(bar_buf, format="png", bbox_inches="tight")
    plt.close(fig_bar)
    bar_buf.seek(0)

    # =========================
    # Savings with Hybrid/Electric
    # =========================
    hybrid_saving = prediction - 100  # assume hybrid emits ~100 g/km
    electric_saving = prediction - 0  # assume EV emits ~0 g/km

    st.subheader(" Potential Savings")
    st.write(f"- If you switch to Hybrid, you save ~**{hybrid_saving:.0f} g/km CO2.")
    st.write(f"- If you switch to Electric, you save ~**{electric_saving:.0f} g/km CO2.")

    # Add to recommendations
    tips.append(f"Switching to hybrid saves ~{hybrid_saving:.0f} g/km CO2.")
    tips.append(f"Switching to electric saves ~{electric_saving:.0f} g/km CO2.")

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

        elements.append(Paragraph("CO2 Emission Report", styles["Title"]))
        elements.append(Spacer(1, 12))

        # Vehicle details
        elements.append(Paragraph("Vehicle Details", styles["Heading2"]))
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
        elements.append(Paragraph("Prediction", styles["Heading2"]))
        elements.append(Paragraph(f"Predicted CO2 Emission: <b>{prediction:.2f} g/km</b>", styles["Normal"]))
        elements.append(Paragraph(f"Emission Status: <b>{status}</b>", styles["Normal"]))
        elements.append(Paragraph(f"Green Score: <b>{green_score:.0f}/100 {badge}</b>", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Gauge Chart
        elements.append(Paragraph("Gauge Chart", styles["Heading2"]))
        chart_img = Image(chart_buf, width=400, height=250)
        elements.append(chart_img)
        elements.append(Spacer(1, 12))

        # Bar Chart
        elements.append(Paragraph("Emission Comparison", styles["Heading2"]))
        bar_img = Image(bar_buf, width=400, height=250)
        elements.append(bar_img)
        elements.append(Spacer(1, 12))

        # Savings
        elements.append(Paragraph("Potential CO2 Savings", styles["Heading2"]))
        elements.append(Paragraph(f"- Hybrid: Save ~{hybrid_saving:.0f} g/km", styles["Normal"]))
        elements.append(Paragraph(f"- Electric: Save ~{electric_saving:.0f} g/km", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Tips
        elements.append(Paragraph("Eco-Friendly Suggestions:", styles["Heading2"]))
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