# 🏏 Hindi Sports Sentiment Analyzer using MuRIL

A web-based **Natural Language Processing (NLP)** application that analyzes **Hindi sports-related text** and classifies its sentiment as **Positive**, **Negative**, or **Neutral** using a **fine-tuned MuRIL (Multilingual Representations for Indian Languages)** transformer model.

The application features a modern frontend built with **Vite + Preact** and a **FastAPI** backend serving the fine-tuned model for real-time sentiment prediction.

---

# 📌 Project Overview

Sentiment analysis for Indian languages remains more challenging than English due to limited high-quality datasets and linguistic diversity. This project focuses specifically on **Hindi sports content**, enabling users to analyze news headlines, match reactions, and sports-related comments.

Instead of using external AI APIs, the application performs inference locally using a fine-tuned **MuRIL** model.

---

# ✨ Features

* Analyze Hindi sports-related text
* Classify sentiment into:

  * 😊 Positive
  * 😐 Neutral
  * 😞 Negative
* Fine-tuned MuRIL transformer model
* FastAPI backend for prediction
* Modern responsive interface using Vite + Preact
* Confidence score for predictions
* Example Hindi sports sentences
* Clean and intuitive user interface

---

# 🛠 Tech Stack

## Frontend

* Vite
* Preact
* JavaScript
* HTML5
* CSS3

## Backend

* FastAPI
* Python
* Uvicorn

## Machine Learning

* PyTorch
* Hugging Face Transformers
* MuRIL (`google/muril-base-cased`)
* Scikit-learn
* Pandas
* NumPy

## Version Control

* Git
* GitHub

---

# 🤖 Model

This project uses Google's **MuRIL (Multilingual Representations for Indian Languages)** transformer model.

Base Model:

```text
google/muril-base-cased
```

The model has been fine-tuned for **three-class sentiment classification**:

* Positive
* Neutral
* Negative

MuRIL is specifically designed for Indian languages, making it well suited for Hindi NLP tasks.

---

# 📂 Dataset

The project is trained using the **SentiHin-2500** Hindi sentiment dataset.

The dataset is processed and prepared for three-class sentiment classification before fine-tuning the transformer model.

---

# ⚙️ Project Workflow

```text
User enters Hindi sports text
            │
            ▼
Frontend (Vite + Preact)
            │
            ▼
FastAPI Backend
            │
            ▼
MuRIL Tokenizer
            │
            ▼
Fine-tuned MuRIL Model
            │
            ▼
Sentiment Prediction
            │
            ▼
Positive / Neutral / Negative + Confidence Score
```

---

# 📁 Project Structure

```text
Hindi-Sports-Sentiment-Analyzer/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── model_loader.py
│   └── prediction.py
│
├── dataset/
│
├── model/
│
├── README.md
│
└── .gitignore
```

---

# 🚀 Installation

## Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Hindi-Sports-Sentiment-Analyzer.git
```

```bash
cd Hindi-Sports-Sentiment-Analyzer
```

---

## Backend Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
uvicorn app:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger API Docs:

```text
http://127.0.0.1:8000/docs
```

---

## Frontend Setup

```bash
cd frontend
```

Install packages:

```bash
npm install
```

Run the application:

```bash
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

# 📡 API

## POST /predict

Example Request

```json
{
  "text": "भारतीय टीम ने शानदार प्रदर्शन किया।"
}
```

Example Response

```json
{
  "sentiment": "Positive",
  "confidence": 0.94
}
```

---

# 💡 Example Predictions

| Input                              | Prediction |
| ---------------------------------- | ---------- |
| भारतीय टीम ने शानदार जीत हासिल की। | Positive   |
| टीम का प्रदर्शन बहुत खराब रहा।     | Negative   |
| मैच कल शाम सात बजे शुरू होगा।      | Neutral    |

---

# 📊 Current Project Status

* ✅ Project finalized
* ✅ Frontend developed
* ✅ GitHub repository created
* ✅ FastAPI backend configured
* ✅ MuRIL model integrated
* ✅ Dataset added
* ⏳ Model fine-tuning in progress
* ⏳ Model evaluation
* ⏳ Frontend and backend integration
* ⏳ Deployment

---

# 🔮 Future Improvements

* Support Hindi-English code-mixed text
* Expand the sports-specific dataset
* Add attention visualization for predictions
* Deploy using Docker
* Compare MuRIL with IndicBERT
* Add batch prediction support
* Support additional Indian languages
* Add prediction history
* Improve handling of sarcasm and figurative language

---

# ⚠️ Limitations

* Optimized primarily for Hindi sports-related text
* Performance depends on the quality and diversity of the training dataset
* Sarcasm and highly contextual statements may be difficult to classify
* Code-mixed text may reduce prediction quality

---

# 📸 Screenshots

Add screenshots here after deployment.

```text
Home Page

Prediction Result

Positive Example

Negative Example
```

---

# 🎯 Learning Outcomes

This project demonstrates practical experience with:

* Natural Language Processing
* Transformer fine-tuning
* Hugging Face Transformers
* MuRIL
* FastAPI
* Vite + Preact
* Model serving
* REST API development
* Git and GitHub
* End-to-end ML application development

---

# 🙏 Acknowledgements

* Google Research (MuRIL)
* Hugging Face
* PyTorch
* FastAPI
* SentiHin Dataset Contributors

---

# 👨‍💻 Author

**Swen Lemos**

B.E. Computer Science Engineering

St. Francis Institute of Technology, Mumbai

---

# 📄 License

This project is intended for educational and research purposes. You may add an MIT License if you plan to make the project open source.
