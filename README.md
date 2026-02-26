# 🧠 SmartDoc AI - RAG Document Assistant

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?logo=google&logoColor=white)

## 📌 Overview
SmartDoc AI is an intelligent document Q&A assistant built with **Python, LangChain, and Google Gemini 3.0 Flash**. It utilizes **Retrieval-Augmented Generation (RAG)** to allow users to upload PDF documents and ask questions about their content in natural language. 

## ✨ Features
* **PDF Processing:** Extracts and chunks text from uploaded PDF files accurately.
* **Vector Search:** Uses FAISS vector database for fast and precise context retrieval.
* **AI-Powered Responses:** Integrates Google Gemini 3.0 Flash for high-quality, context-aware answers.
* **Interactive UI:** Built with Streamlit for a seamless, responsive, and user-friendly chat experience.
* **Secure by Design:** Users provide their own API keys via the UI, ensuring data privacy.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **LLM & Embeddings:** Google Gemini 3.0 API
* **Framework:** LangChain
* **Vector Store:** FAISS
* **PDF Parsing:** PyPDFLoader

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ahmetyusgencer/smartdoc-ai.git](https://github.com/ahmetyusgencer/smartdoc-ai.git)
   cd smartdoc-ai
