import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
from model import compute_score

retention_model = xgb.XGBClassifier()
retention_model.load_model('ipl_retention_model.json')

salary_model = xgb.XGBRegressor()
salary_model.load_model('ipl_salary_prediction_model.json')

st.title("🏏 IPL Player Retention Predictor")

st.markdown("Enter your performance details below (from last season):")

total_runs = st.number_input("Total Runs Scored", min_value=0, value=300)
balls_faced = st.number_input("Balls Faced", min_value=0, value=250)
total_wickets = st.number_input("Total Wickets Taken", min_value=0, value=5)
total_runs_conceded = st.number_input("Total Runs Conceded (Bowling)", min_value=0, value=200)
balls_bowled = st.number_input("Balls Bowled", min_value=0, value=120)

strike_rate = (total_runs / balls_faced) * 100 if balls_faced > 0 else 0
economy_rate = (total_runs_conceded / balls_bowled) * 6 if balls_bowled > 0 else 0

st.write(f"📈 Calculated Strike Rate: **{strike_rate:.2f}**")
st.write(f"🎯 Calculated Economy Rate: **{economy_rate:.2f}**")

if st.button("Predict Retention"):
    input_data = np.array([[total_runs, balls_faced, strike_rate,
                            total_wickets, total_runs_conceded,
                            balls_bowled, economy_rate]])

    prediction = retention_model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Congratulations! You are likely to be *retained*.")
        
        input_df = pd.DataFrame(input_data, columns=[
            'total_runs', 'balls_faced', 'strike_rate',
            'total_wickets', 'total_runs_conceded',
            'balls_bowled', 'economy_rate'
        ])

        score = compute_score(input_df.iloc[0])
        salary = salary_model.predict(np.array([[score]]))[0]
        st.write(f"💰 Estimated Salary: **₹{salary:.2f} Lakhs**")
    else:
        st.error("❌ Sorry! You are likely *not retained* based on your stats.")

        improvement_tips = []

        if total_runs <= 300 and not (total_runs > 150 and strike_rate > 140):
            if total_runs <= 150:
                improvement_tips.append("- Try to score more than **150 runs**.")
            if strike_rate <= 140:
                improvement_tips.append("- Improve your **strike rate** above **140**.")

        if total_wickets <= 10 and not (total_wickets > 5 and economy_rate < 8):
            if total_wickets <= 5:
                improvement_tips.append("- Aim to take more than **5 wickets**.")
            if economy_rate >= 8 and balls_bowled > 0:
                improvement_tips.append("- Reduce your **economy rate** below **8.0**.")

        if not improvement_tips:
            improvement_tips.append("- Try to improve your overall performance metrics.")

        st.markdown("### 📌 How to Improve:")
        for tip in improvement_tips:
            st.markdown(tip)
        
    st.markdown("📌 Note: This prediction is based on XGBoost trained on IPL data from seasons 2022–2024.")
