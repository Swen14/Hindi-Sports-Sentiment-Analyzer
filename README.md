# 🏏 Hindi Sports Sentiment Analyzer using MuRIL

A web-based **Natural Language Processing (NLP)** application that analyzes **Hindi sports-related text** and classifies its sentiment as **Positive**, **Negative**, or **Neutral** using a fine-tuned **MuRIL (Multilingual Representations for Indian Languages)** transformer model.

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

The model has been fine-tuned for three-class sentiment classification:

* Positive
* Neutral
* Negative

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
├── backend/
├── dataset/
├── model/
├── README.md
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

## Backend

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
uvicorn app:app --reload
```

Backend

```text
http://127.0.0.1:8000
```

API Docs

```text
http://127.0.0.1:8000/docs
```

---

## Frontend

```bash
cd frontend
```

```bash
npm install
```

```bash
npm run dev
```

Frontend

```text
http://localhost:5173
```

---

# 📡 API

## POST `/predict`

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

# 📸 Screenshots

> Add screenshots after completing the project.

### 1. Home Page

```
screenshots/home-page.png
```

### 2. Positive Prediction

```
screenshots/positive-prediction.png
```

### 3. Negative Prediction

```
screenshots/negative-prediction.png
```

### 4. Neutral Prediction

```
screenshots/neutral-prediction.png
```

### 5. Loading State

```
screenshots/loading-state.png
```

### 6. Mobile View (Optional)

```
screenshots/mobile-view.png
```

Example Markdown:

```markdown
## Screenshots

### Home Page

![Home Page](screenshots/home-page.png)

### Positive Prediction

![Positive Prediction](screenshots/positive-prediction.png)

### Negative Prediction

![Negative Prediction](screenshots/negative-prediction.png)

### Neutral Prediction

![Neutral Prediction](screenshots/neutral-prediction.png)

### Loading State

![Loading State](screenshots/loading-state.png)
```

---

# 👨‍💻 Author

**Swen Lemos**

B.E. Computer Science Engineering

St. Francis Institute of Technology, Mumbai

---

# 📄 License

This project is intended for educational and research purposes.
