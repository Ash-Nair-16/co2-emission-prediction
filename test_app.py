import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import patches
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# =========================
# Helpers: Matplotlib Semi-Gauge & EV/Hybrid Chart (for PDF)
# =========================
def generate_static_semi_gauge(value, avg, safe_limit=150, moderate_limit=250, max_val=400,
                               figsize=(6,3), title="CO₂ Emission Gauge"):
    """
    Draws a semi-circular gauge (speedometer-like) using cartesian Wedge patches so
    it resembles the Plotly gauge, then returns PNG buffer.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')

    # center and radius
    center = (0.0, 0.0)
    R = 1.0

    # angles in degrees: 180 (left) -> 0 (right)
    def val_to_deg(v):
        return 180.0 - (v / max_val) * 180.0

    theta_safe_start = val_to_deg(safe_limit)    # angle at safe_limit
    theta_moderate_start = val_to_deg(moderate_limit)

    # Wedges: green (safe), orange (moderate), red (high)
    wedge_green = patches.Wedge(center, R, theta_safe_start, 180.0, facecolor='green', alpha=0.7, edgecolor='none')
    wedge_orange = patches.Wedge(center, R, theta_moderate_start, theta_safe_start, facecolor='orange', alpha=0.75, edgecolor='none')
    wedge_red = patches.Wedge(center, R, 0.0, theta_moderate_start, facecolor='red', alpha=0.75, edgecolor='none')

    ax.add_patch(wedge_green)
    ax.add_patch(wedge_orange)
    ax.add_patch(wedge_red)

    # Outer ring for nicer look
    outer = patches.Wedge(center, R * 1.02, 0.0, 180.0, facecolor='none', edgecolor='lightgray', lw=1)
    ax.add_patch(outer)

    # Draw ticks (0, safe_limit, moderate_limit, max_val)
    ticks = [0, safe_limit, moderate_limit, max_val]
    for t in ticks:
        deg = np.deg2rad(val_to_deg(t))
        x = 0.9 * R * np.cos(deg)
        y = 0.9 * R * np.sin(deg)
        ax.text(x, y - 0.06, f"{int(t)}", horizontalalignment='center', verticalalignment='center', fontsize=9)

    # Needle for prediction (black)
    angle_pred_deg = val_to_deg(value)
    angle_pred_rad = np.deg2rad(angle_pred_deg)
    x_end = 0.75 * R * np.cos(angle_pred_rad)
    y_end = 0.75 * R * np.sin(angle_pred_rad)
    ax.plot([0, x_end], [0, y_end], color='black', lw=3, zorder=5)
    ax.scatter([0], [0], color='black', s=40, zorder=6)

    # Marker for class average (blue triangle on rim)
    angle_avg_deg = val_to_deg(avg)
    angle_avg_rad = np.deg2rad(angle_avg_deg)
    x_avg = 0.92 * R * np.cos(angle_avg_rad)
    y_avg = 0.92 * R * np.sin(angle_avg_rad)
    ax.scatter([x_avg], [y_avg], marker=(3, 0, -angle_avg_deg+90), color='blue', s=80, zorder=6)  # triangle marker rotated

    # Title and values
    ax.text(0, -0.35, f"{title}", horizontalalignment='center', fontsize=12, weight='bold')
    ax.text(0, -0.52, f"Your car: {value:.1f} g/km   |   Class avg: {avg:.1f} g/km", horizontalalignment='center', fontsize=10)

    # Set limits so it's centered
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.6, 1.05)

    # Save to buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_ev_comparison_chart_pdf(car_val, hybrid_val=100, ev_val=0, figsize=(5,3)):
    """
    Creates a clear bar chart comparing 'Your Car', 'Hybrid' and 'EV' and returns PNG buffer.
    """
    fig, ax = plt.subplots(figsize=figsize)
    categories = ["Your Car", "Hybrid", "EV"]
    values = [car_val, hybrid_val, ev_val]
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # red, orange, green

    bars = ax.bar(categories, values, color=colors, alpha=0.9)
    ax.set_ylabel("CO₂ Emissions (g/km)")
    ax.set_ylim(0, max(max(values) * 1.2, 300))
    ax.set_title("CO₂: Your Car vs Hybrid vs EV")

    # add value labels
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + (ax.get_ylim()[1] * 0.03), f"{h:.1f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# =========================
# Load model & dataset (ensure model is pipeline)
# =========================
model = joblib.load("best_co2_model.pkl")  # should be pipeline including preprocessing
df = pd.read_csv("co2 Emissions.csv")

st.set_page_config(page_title="🌍 CO₂ Emission Predictor", layout="wide")

# =========================
# Header & Inputs
# =========================
st.title("🌍 CO₂ Emission Predictor")
st.write("Enter vehicle details to estimate **CO₂ emissions (g/km)**, see a gauge and download a PDF report.")

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
# Predict & Display
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

    # predict (pipeline handles preprocessing)
    prediction = model.predict(input_df)[0]

    # class average
    avg_class_emission = df[df["Vehicle Class"] == vehicle_class]["CO2 Emissions(g/km)"].mean()

    # classify
    if prediction < 150:
        status = "✅ Safe"
        color = "green"
        tips = ["Keep driving efficiently!", "Maintain your vehicle regularly."]
    elif prediction <= 250:
        status = "⚠️ Moderate"
        color = "orange"
        tips = ["Avoid aggressive driving.", "Use cruise control when possible.", "Reduce extra weight in the car."]
    else:
        status = "❌ High"
        color = "red"
        tips = ["Consider hybrid/electric vehicles.", "Carpool or use public transport.", "Limit AC usage when not needed."]

    # show numeric results
    st.markdown(f"## Estimated CO₂ Emission: <span style='color:{color}'>{prediction:.1f} g/km</span>", unsafe_allow_html=True)
    st.markdown(f"### Emission Level: **{status}**")

    # Plotly gauge (interactive)
    st.subheader("📊 CO₂ Emission Gauge")
    safe_limit = 150
    moderate_limit = 250
    max_val = 400

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        title={'text': "CO₂ Emissions (g/km)"},
        delta={'reference': avg_class_emission, 'relative': False, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, safe_limit], 'color': "green"},
                {'range': [safe_limit, moderate_limit], 'color': "orange"},
                {'range': [moderate_limit, max_val], 'color': "red"}
            ],
            'threshold': {'line': {'color': "blue", 'width': 4}, 'thickness': 0.75, 'value': avg_class_emission}
        }
    ))
    fig.update_layout(height=380, margin=dict(l=40, r=40, t=80, b=40), font=dict(size=16))
    st.plotly_chart(fig, use_container_width=True)

    # Green score
    st.subheader("🌱 Green Score")
    green_score = max(0, 100 - (prediction / max_val) * 100)
    if green_score >= 80:
        score_status = "🌟 Excellent! Your car is very eco-friendly."
        score_color = "green"
    elif green_score >= 50:
        score_status = "⚖️ Average. There’s room for improvement."
        score_color = "orange"
    else:
        score_status = "🔴 Poor. High environmental impact."
        score_color = "red"

    st.progress(int(green_score))
    st.markdown(f"### Green Score: <span style='color:{score_color}'>{green_score:.1f} / 100</span>", unsafe_allow_html=True)
    st.write(score_status)

    # EV/Hybrid savings
    st.subheader("⚡ Potential Savings with EV/Hybrid")
    savings_ev = max(0, prediction - 0)
    savings_hybrid = max(0, prediction - 100)
    st.write(f"If you switched to an **Electric Vehicle (EV)**, you could save about **{savings_ev:.1f} g/km** of CO₂.")
    st.write(f"If you switched to a **Hybrid Vehicle**, you could save about **{savings_hybrid:.1f} g/km** of CO₂.")

    # =========================
    # Downloadable PDF (uses Matplotlib static images)
    # =========================
    st.subheader("📥 Download Your PDF Report")

    def generate_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        # Title and details
        elements.append(Paragraph(" CO2 Emission Report", styles["Title"]))
        elements.append(Spacer(1, 8))

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
        elements.append(Spacer(1, 8))

        # Prediction + status
        elements.append(Paragraph("🔮 Prediction", styles["Heading2"]))
        elements.append(Paragraph(f"Predicted CO2 Emission: <b>{prediction:.1f} g/km</b>", styles["Normal"]))
        elements.append(Paragraph(f"Emission Status: <b>{status}</b>", styles["Normal"]))
        elements.append(Spacer(1, 8))

        # Green score
        elements.append(Paragraph(" Green Score", styles["Heading2"]))
        elements.append(Paragraph(f"Score: <b>{green_score:.1f} / 100</b>", styles["Normal"]))
        elements.append(Paragraph(score_status, styles["Normal"]))
        elements.append(Spacer(1, 8))

        # Semi-gauge (Matplotlib static version)
        elements.append(Paragraph(" Emission Gauge", styles["Heading2"]))
        gauge_buf = generate_static_semi_gauge(prediction, avg_class_emission, safe_limit, moderate_limit, max_val)
        elements.append(Image(gauge_buf, width=450, height=220))
        elements.append(Spacer(1, 8))

        # EV/Hybrid savings text
        elements.append(Paragraph(" Potential Savings", styles["Heading2"]))
        elements.append(Paragraph(f"If you switched to an <b>Electric Vehicle (EV)</b>, you could save about <b>{savings_ev:.1f} g/km</b> of CO2.", styles["Normal"]))
        elements.append(Paragraph(f"If you switched to a <b>Hybrid Vehicle</b>, you could save about <b>{savings_hybrid:.1f} g/km</b> of CO2.", styles["Normal"]))
        elements.append(Spacer(1, 8))

        # EV/Hybrid comparison chart (Matplotlib)
        elements.append(Paragraph(" EV / Hybrid Comparison", styles["Heading2"]))
        evchart_buf = generate_ev_comparison_chart_pdf(prediction, hybrid_val=100, ev_val=0)
        elements.append(Image(evchart_buf, width=450, height=260))
        elements.append(Spacer(1, 8))

        # Tips
        elements.append(Paragraph(" Eco-Friendly Suggestions:", styles["Heading2"]))
        for t in tips:
            elements.append(Paragraph(f"- {t}", styles["Normal"]))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf_buffer = generate_pdf()
    st.download_button(label="⬇️ Download Report (PDF)", data=pdf_buffer, file_name="co2_emission_report.pdf", mime="application/pdf")
