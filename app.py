import sys
sys.modules['torch.classes'] = None
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import os
import requests
import traceback

os.environ["USE_TF"] = "0"  # Force PyTorch mode

# === Custom LLM Call: Patient CSV + Query Context ===
def generate_response_from_patient_csv(df, query):
    # Summarize context from the data
    total_days = df.shape[0]
    avg_admissions = df['y'].mean()
    max_day = df.loc[df['y'].idxmax()]
    recent = df.tail(7)

    context = f"""
Data Summary:
- Total days: {total_days}
- Average daily admissions: {avg_admissions:.2f}
- Highest admission: {max_day['y']} on {max_day['ds'].date()}

Recent 7 Days:
{recent.to_string(index=False)}
"""

    full_prompt = f"""### Instruction:
Use the patient admission data below to answer the user's question.

### Context:
{context}

### Question:
{query}

### Response:"""

    headers = {
        "Authorization": f"Bearer {st.secrets['HF_API_KEY']}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.3
        }
    }

    response = requests.post(
        "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",  # Public supported model
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        try:
            outputs = response.json()
            if isinstance(outputs, list) and "generated_text" in outputs[0]:
                return outputs[0]["generated_text"]
            else:
                return "⚠️ No proper output returned."
        except ValueError:
            return "⚠️ Response is not in JSON format."
    else:
        # Enhanced error handling
        print("\n[HF API ERROR]")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        print(f"Request URL: https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta")
        masked_key = headers['Authorization'][:10] + '...' if 'Authorization' in headers else 'None'
        print(f"Headers: {{'Authorization': '{masked_key}', 'Content-Type': '{headers.get('Content-Type', '')}'}}")
        print(f"Payload: {payload}")
        print(f"Model ID: HuggingFaceH4/zephyr-7b-beta")
        error_message = response.text if response.text else "No error message provided."
        return f"❌ Error: {response.status_code} - {error_message}"

# === UI SETUP ===
st.set_page_config(page_title="AI Hospital Assistant", layout="wide")
st.title("🏥 AI-Powered Hospital Staffing + RAG Assistant")

# === File Upload ===
st.markdown("### 📂 Upload Patient Admission Data")
uploaded_file = st.file_uploader("Upload your CSV file (columns: d, y)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom file uploaded!")
else:
    st.info("Using default data from `default_data/patient_data.csv`")
    df = pd.read_csv("default_data/patient_data.csv")

# Ensure the columns are named correctly
df.columns = ['ds', 'y']  # Rename columns for consistency
df['ds'] = pd.to_datetime(df['ds'])  # Convert 'ds' to datetime

# === Forecasting ===
st.subheader("📊 Uploaded / Default Data Preview")
st.write(df.head())

model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

st.subheader("📈 Forecasted Patient Admissions")
st.line_chart(forecast[['ds', 'yhat']].set_index('ds'))

forecast['doctors_required'] = np.ceil(forecast['yhat'] / 10)
forecast['nurses_required'] = forecast['doctors_required'] * 2
forecast['support_staff_required'] = np.ceil(forecast['yhat'] / 20)

st.subheader("👩‍⚕️ Required Staff Forecast")
st.write(forecast[['ds', 'yhat', 'doctors_required', 'nurses_required', 'support_staff_required']].tail(10))

st.subheader("📌 Staff Allocation Overview")
st.bar_chart(forecast[['ds', 'doctors_required', 'nurses_required', 'support_staff_required']].set_index('ds'))

st.subheader("⏳ Shift Schedule for Next 7 Days")
shifts = []
for _, row in forecast.tail(7).iterrows():
    day = row['ds'].strftime('%Y-%m-%d')
    shifts.append({
        "Date": day,
        "Morning Shift": f"{int(row['doctors_required']/2)} Doctors, {int(row['nurses_required']/2)} Nurses, {int(row['support_staff_required']/2)} Support Staff",
        "Night Shift": f"{int(row['doctors_required']/2)} Doctors, {int(row['nurses_required']/2)} Nurses, {int(row['support_staff_required']/2)} Support Staff"
    })
st.write(pd.DataFrame(shifts))

# === Default Questions Section ===
st.markdown("---")
st.subheader("🧠 Default Questions About Patient Admissions")

# Define default questions
default_questions = {
    "General Information": [
        "How many patients were admitted this week?",
        "What is the average number of admissions?"
    ],
    "Peak Load": [
        "What was the peak load so far?",
        "On which day did we have the highest admissions?"
    ],
    "Staffing Needs": [
        "How many doctors are needed tomorrow?",
        "How should staff be scheduled for the next 7 days?"
    ],
    "Recent Trends": [
        "What were the admissions like in the last 7 days?",
        "How do the admissions compare to last month?"
    ],
    "Future Projections": [
        "What are the projected admissions for the next month?",
        "How many support staff will be needed next week?"
    ]
}

# Display questions as dropdown menus
for category, questions in default_questions.items():
    with st.expander(category):
        for question in questions:
            if st.button(question):
                st.session_state.user_input = question  # Store the question in session state
                st.session_state.input_field = question  # Set the input field value

# Input field for user questions
if 'user_input' in st.session_state:
    user_input = st.session_state.user_input
else:
    user_input = ""

# Create the input field without a default value
input_value = st.text_input("Type your question here:", value=user_input, key="input_field")

if input_value:
    query = input_value
    with st.spinner("Retrieving answer..."):
        try:
            response = generate_response_from_patient_csv(df, query)
            st.success(response)
        except Exception as e:
            print("[EXCEPTION] Error retrieving answer:")
            traceback.print_exc()
            st.error(f"❌ Error retrieving answer: {str(e)}")

st.markdown("---")
st.caption("Built with Streamlit, Prophet, LangChain, Hugging Face")
