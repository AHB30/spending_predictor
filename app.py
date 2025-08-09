import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Project Title
st.title("📊 ML-Powered Income & Expense Predictor")

# Upload CSV
uploaded_file = st.file_uploader("Upload your income/expense CSV", type="csv")

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data")
    st.dataframe(df)

    # Add MonthIndex for ML models
    df["MonthIndex"] = range(1, len(df) + 1)

    # The Mind of Prediction
    # -------------------- Income Prediction --------------------
    income_model = LinearRegression()
    X_income = df[["MonthIndex"]]
    y_income = df["Income"]
    income_model.fit(X_income, y_income)

    predicted_income = income_model.predict([[len(df) + 1]])[0]

    # -------------------- Expense Prediction --------------------
    expense_model = LinearRegression()
    y_expense = df["Expenses"]
    expense_model.fit(X_income, y_expense)

    predicted_expense = expense_model.predict([[len(df) + 1]])[0]

    # -------------------- Spending Insights --------------------
    df["Balance"] = df["Income"] - df["Expenses"]
    max_spend = df.loc[df["Expenses"].idxmax()]
    min_spend = df.loc[df["Expenses"].idxmin()]

    st.subheader("📈 Spending Insights")
    st.write(f"**Highest Spending:** {max_spend['Month']} – ${max_spend['Expenses']}")
    st.write(f"**Lowest Spending:** {min_spend['Month']} – ${min_spend['Expenses']}")

    # -------------------- Income Insights --------------------
    max_income = df.loc[df["Income"].idxmax()]
    min_income = df.loc[df["Income"].idxmin()]

    st.subheader("📥 Income Insights")
    st.write(f"**Highest Income:** {max_income['Month']} – ${max_income['Income']}")
    st.write(f"**Lowest Income:** {min_income['Month']} – ${min_income['Income']}")

    # -------------------- Charts --------------------
    st.subheader("📊 Expense Trend")
    fig1, ax1 = plt.subplots()
    ax1.plot(df["Month"], df["Expenses"], marker="o", label="Actual Expense")
    ax1.plot("Next", predicted_expense, marker="x", color="red", label="Predicted")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Expenses")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("📈 Income Trend")
    fig2, ax2 = plt.subplots()
    ax2.plot(df["Month"], df["Income"], marker="o", label="Actual Income")
    ax2.plot("Next", predicted_income, marker="x", color="green", label="Predicted")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Income")
    ax2.legend()
    st.pyplot(fig2)

    # -------------------- Prediction Summary --------------------
    st.subheader("🔮 Next Month Forecast")
    st.write(f"💰 Estimated Income: **${predicted_income:.2f}**")
    st.write(f"💸 Estimated Expense: **${predicted_expense:.2f}**")

    balance = predicted_income - predicted_expense
    st.write(f"🧾 Projected Balance: **${balance:.2f}**")

else:
    st.info("📁 Please upload a CSV file to begin.")
