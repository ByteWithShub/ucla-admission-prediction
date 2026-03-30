#Streamlit app for UCLA Admission Prediction

import streamlit as st
import joblib
import pandas as pd

from src.preprocessing import encode_data

st.set_page_config(
    page_title="UCLA Admission Prediction",
    layout="wide"
)

# Load saved artifacts
model = joblib.load("models/ucla_mlp_model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")

# Try loading loss curve data
try:
    loss_df = pd.read_csv("outputs/loss_curve.csv")
except FileNotFoundError:
    loss_df = None


#Helper functions for prediction and insights

def prepare_input_df(gre_score, toefl_score, university_rating, sop, lor, cgpa, research):
    input_data = {
        "GRE_Score": gre_score,
        "TOEFL_Score": toefl_score,
        "University_Rating": university_rating,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "Research": research,
        "Admit_Chance": 1
    }

    input_df = pd.DataFrame([input_data])
    input_df["University_Rating"] = input_df["University_Rating"].astype("object")
    input_df["Research"] = input_df["Research"].astype("object")

    input_df = encode_data(input_df)
    input_df = input_df.drop("Admit_Chance", axis=1)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    return input_df


def predict_admission(gre_score, toefl_score, university_rating, sop, lor, cgpa, research):
    input_df = prepare_input_df(
        gre_score, toefl_score, university_rating, sop, lor, cgpa, research
    )
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return prediction, probability


def profile_strength_text(gre_score, toefl_score, cgpa, research):
    score = 0

    if gre_score >= 320:
        score += 1
    if toefl_score >= 105:
        score += 1
    if cgpa >= 8.5:
        score += 1
    if research == 1:
        score += 1

    if score >= 4:
        return "This looks like a strong academic profile."
    elif score >= 2:
        return "This profile looks moderately competitive."
    else:
        return "This profile may need improvement to become more competitive."


def improvement_suggestions(gre_score, toefl_score, sop, lor, cgpa, research):
    suggestions = []

    if gre_score < 320:
        suggestions.append("Consider improving the GRE score to strengthen the application.")
    if toefl_score < 105:
        suggestions.append("A higher TOEFL score may improve the overall profile.")
    if cgpa < 8.5:
        suggestions.append("A stronger CGPA can significantly improve admission competitiveness.")
    if sop < 4.0:
        suggestions.append("A stronger Statement of Purpose may improve the application quality.")
    if lor < 4.0:
        suggestions.append("Stronger recommendation letters could help the application stand out.")
    if research == 0:
        suggestions.append("Adding research experience may improve admission chances.")

    if not suggestions:
        suggestions.append("This profile is already strong across the major academic indicators.")

    return suggestions


#Heading and description

st.title("UCLA Admission Prediction Studio")
st.markdown(
    """
    This app predicts whether a student has a **high chance of admission** using a  
    **Neural Network (MLPClassifier)** trained on academic profile data.

    The original notebook has been converted into a modular Python project with model saving,
    preprocessing, logging, and Streamlit deployment support.
    """
)

tab1, tab2, tab3 = st.tabs(["Predict Admission", " What-If Lab", "Model Insights"])

#Prediction tab

with tab1:
    st.subheader("Student Profile")

    left, right = st.columns([1.2, 1])

    with left:
        gre_score = st.slider("GRE Score", 260, 340, 320)
        toefl_score = st.slider("TOEFL Score", 0, 120, 105)
        university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
        sop = st.slider("SOP Strength", 1.0, 5.0, 3.0, step=0.5)
        lor = st.slider("LOR Strength", 1.0, 5.0, 3.0, step=0.5)
        cgpa = st.slider("CGPA", 0.0, 10.0, 8.0, step=0.1)
        research = st.selectbox("Research Experience", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        predict_button = st.button("Predict Admission Category")

    with right:
        st.markdown("### Profile Snapshot")
        c1, c2 = st.columns(2)
        c1.metric("GRE", gre_score)
        c2.metric("TOEFL", toefl_score)

        c3, c4 = st.columns(2)
        c3.metric("CGPA", cgpa)
        c4.metric("Research", "Yes" if research == 1 else "No")

        c5, c6, c7 = st.columns(3)
        c5.metric("Univ Rating", university_rating)
        c6.metric("SOP", sop)
        c7.metric("LOR", lor)

        st.markdown("### Quick Read")
        st.info(profile_strength_text(gre_score, toefl_score, cgpa, research))

    if predict_button:
        prediction, probability = predict_admission(
            gre_score, toefl_score, university_rating, sop, lor, cgpa, research
        )

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("High Chance of Admission")
        else:
            st.error("Lower Chance of Admission")

        st.info(f"Predicted probability of high admission chance: {probability:.2%}")

        st.markdown("### Recommendations")
        for item in improvement_suggestions(gre_score, toefl_score, sop, lor, cgpa, research):
            st.write(f"- {item}")

#What-if lab tab

with tab2:
    st.subheader("What-If Lab")
    st.write(
        "Use this section to explore how changes in GRE, TOEFL, CGPA, SOP, LOR, or research experience may affect the prediction."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Original Profile")
        base_gre = st.slider("Base GRE", 260, 340, 315, key="base_gre")
        base_toefl = st.slider("Base TOEFL", 0, 120, 100, key="base_toefl")
        base_univ = st.selectbox("Base University Rating", [1, 2, 3, 4, 5], key="base_univ")
        base_sop = st.slider("Base SOP", 1.0, 5.0, 3.0, step=0.5, key="base_sop")
        base_lor = st.slider("Base LOR", 1.0, 5.0, 3.0, step=0.5, key="base_lor")
        base_cgpa = st.slider("Base CGPA", 0.0, 10.0, 8.0, step=0.1, key="base_cgpa")
        base_research = st.selectbox("Base Research", [0, 1], key="base_research", format_func=lambda x: "Yes" if x == 1 else "No")

    with col_b:
        st.markdown("### Improved Scenario")
        new_gre = st.slider("New GRE", 260, 340, 325, key="new_gre")
        new_toefl = st.slider("New TOEFL", 0, 120, 108, key="new_toefl")
        new_univ = st.selectbox("New University Rating", [1, 2, 3, 4, 5], key="new_univ")
        new_sop = st.slider("New SOP", 1.0, 5.0, 4.0, step=0.5, key="new_sop")
        new_lor = st.slider("New LOR", 1.0, 5.0, 4.0, step=0.5, key="new_lor")
        new_cgpa = st.slider("New CGPA", 0.0, 10.0, 8.8, step=0.1, key="new_cgpa")
        new_research = st.selectbox("New Research", [0, 1], key="new_research", format_func=lambda x: "Yes" if x == 1 else "No")

    if st.button("Compare Scenarios"):
        base_pred, base_prob = predict_admission(
            base_gre, base_toefl, base_univ, base_sop, base_lor, base_cgpa, base_research
        )

        new_pred, new_prob = predict_admission(
            new_gre, new_toefl, new_univ, new_sop, new_lor, new_cgpa, new_research
        )

        r1, r2 = st.columns(2)

        with r1:
            st.markdown("### Original Result")
            st.write(f"Prediction: {'High Chance' if base_pred == 1 else 'Lower Chance'}")
            st.write(f"Probability: {base_prob:.2%}")

        with r2:
            st.markdown("### Improved Result")
            st.write(f"Prediction: {'High Chance' if new_pred == 1 else 'Lower Chance'}")
            st.write(f"Probability: {new_prob:.2%}")

        diff = new_prob - base_prob
        if diff > 0:
            st.success(f"The updated profile improved the predicted probability by {diff:.2%}.")
        elif diff < 0:
            st.warning(f"The updated profile reduced the predicted probability by {abs(diff):.2%}.")
        else:
            st.info("Both profiles produced the same probability.")

#Model insights tab

with tab3:
    st.subheader("Model Insights")

    st.markdown("### Project Summary")
    st.write(
        """
        - **Model Used:** MLPClassifier
        - **Task Type:** Binary Classification
        - **Target Rule:** `Admit_Chance >= 0.8` becomes class 1
        - **Preprocessing:** One-hot encoding + MinMax scaling
        """
    )

    st.markdown("### Why this matters")
    st.write(
        """
        This tab helps during demo and presentation because it shows the methodology behind the app,
        not just the output. It connects your code, training logic, and user-facing interface.
        """
    )

    if loss_df is not None:
        st.markdown("### Training Loss Curve")
        st.line_chart(loss_df["Loss"])
    else:
        st.warning("Run `python main.py` first to generate the loss curve file.")

st.markdown("---")
st.caption("Built with Streamlit, scikit-learn, and a modular neural network pipeline.")
