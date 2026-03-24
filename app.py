import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -------------------------
# Load model and scaler
# -------------------------

model = pickle.load(open('model/kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

# -------------------------
# Segment mapping (แก้ให้ตรง logic ธุรกิจ)
# -------------------------

segment_names = {
    0: "Loyal",
    1: "Premium",
    2: "Normal",
    3: "Young"
}

# ลำดับสำหรับแสดงผล (ตรง Power BI)
segment_order = ["Loyal", "Premium", "Normal", "Young"]

# -------------------------
# App title
# -------------------------

st.title("🧠 Customer Segmentation App")

# -------------------------
# Create two columns
# -------------------------

col1, col2 = st.columns([3, 2])

# -------------------------
# LEFT SIDE (Visualization)
# -------------------------

with col1:

    st.header("📊 Customer Segment Distribution")

    try:
        df = pd.read_csv("dataset/segmented_customers.csv")

        # Map segment name
        df['Segment_Name'] = df['Segment'].map(segment_names)

        # Plot
        fig, ax = plt.subplots()

        df['Segment_Name'].value_counts().reindex(segment_order).plot(
            kind='bar',
            ax=ax
        )

        ax.set_xlabel("Segment")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Customer Count by Segment")

        st.pyplot(fig)

        # 🔥 Debug (ดูค่าเฉลี่ยแต่ละกลุ่ม)
        with st.expander("🔍 Cluster Profile (Debug)"):
            st.write(df.groupby('Segment_Name').mean())

    except FileNotFoundError:
        st.warning("Dataset not found.")

# -------------------------
# RIGHT SIDE (Prediction)
# -------------------------

with col2:

    st.header("🔍 Predict Segment")

    with st.form("predict_form"):

        income = st.number_input("Income", min_value=0)
        kids = st.slider("Number of Kids", 0, 3)
        teens = st.slider("Number of Teens", 0, 3)
        recency = st.number_input("Recency", min_value=0)
        wines = st.number_input("Monthly Wine Spend", min_value=0)
        fruits = st.number_input("Monthly Fruit Spend", min_value=0)

        submitted = st.form_submit_button("Predict Segment")

        if submitted:

            # ตรวจ input
            if income == 0 or recency == 0:
                st.warning("Please fill in all required fields")
            else:

                data = [[income, kids, teens, recency, wines, fruits]]
                data_scaled = scaler.transform(data)
                segment = model.predict(data_scaled)

                segment_label = segment_names.get(segment[0], "Unknown")

                st.success(
                    f"Predicted Customer Segment: {segment_label} ({segment[0]})"
                )
