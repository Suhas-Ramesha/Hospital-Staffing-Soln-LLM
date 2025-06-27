# 🏥 AI-Powered Hospital Staffing Assistant


This project is an AI-powered assistant built using **Streamlit** and a **fine-tuned LLM on Hugging Face**, designed to answer questions based on uploaded patient admission data.

---

## 📌 Features

- 🧠 **LLM-backed Natural Language Q&A**
- 📈 Upload your hospital’s patient admission CSV
- 🤖 Automatically generates context and queries your own fine-tuned model
- 🔒 Secure API key management with `secrets.toml`

---

## 📊 Example Use Cases

- “How many patients were admitted last week?”
- “What’s the busiest day in the uploaded data?”
- “Give a summary of admissions over the last 7 days.”

---

## 🚀 Try the App

▶ **[Live Demo Here](https://hospital-staffing-soln-llm-sxknrpw6gnukrdat2trpzq.streamlit.app/)**

---

## 🛠️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hospital-staffing-assistant.git
cd hospital-staffing-assistant
```

### 2. Set up secrets
Create a file .streamlit/secrets.toml and add your Hugging Face API token:

toml
```Copy
Edit
[secrets]
HF_API_KEY = "hf_your_actual_token_here"
```
⚠️ Do NOT commit this file to GitHub. Add it to .gitignore.

### 3. Install dependencies
bash
```Copy
Edit
pip install -r requirements.txt
```

### 4. Run the app
bash
```Copy
Edit
streamlit run app.py
```

## 📁 File Structure
bash
```Copy
Edit
├── app.py                   # Streamlit app file
├── README.md
├── requirements.txt
└── .streamlit/
    └── secrets.toml         # Your Hugging Face API key (not to be committed)
```

## 🤖 Model Details
- Base Model: unsloth/mistral-7b-bnb-4bit

- Fine-tuned on: Patient admission Q&A (custom synthetic + real data)

- Deployed to: [Hugging Face Hub](https://huggingface.co/Suhas319/staffing-assistant-llm) 

## 🙋‍♂️ Author
Suhas Ramesha
🔗 [LinkedIn](https://linkedin.com/suhas-ramesha/)
💻 Built with 💙 using Unsloth, Streamlit, and Hugging Face

## 📄 License
This project is licensed under the MIT License. See LICENSE for details.

### ✅ To Use It:

1. Create a file named `README.md` in your GitHub repo
2. Paste this content in
3. Replace `your-username` in the Git clone link if needed
4. Optionally, adjust model details or add badges
