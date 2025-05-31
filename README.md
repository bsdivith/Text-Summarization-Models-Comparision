# AI-Powered Document Analyzer Using Groq-Hosted LLMs

This project is an interactive Streamlit web application that leverages multiple Large Language Models (LLMs) hosted on the Groq platform to extract, summarize, and analyze content from PDF documents and web URLs. It enables intelligent summarization and question answering (Q&A) across various content types.

---

## 🔍 Features

- **PDF and URL Input Support**  
  Extract text content from local PDF documents or publicly accessible websites.

- **Summarization Engine**  
  Generate summaries with customizable length and focus using models such as:
  - LLaMA 3.3 70B Versatile
  - Gemma 2 9B Instruct
  - LLaMA 3.1 8B Instant

- **Question Answering Module**  
  Ask context-aware questions on the extracted content and receive precise, model-generated answers.

- **Model Selection and Comparison**  
  Choose from a curated set of models with guidance on best use cases for each.

- **User Interface**  
  Built with Streamlit, supporting real-time interaction and model output visualization.

---

## 📌 Project Objectives

- Provide an accessible interface for document intelligence using LLMs.
- Enable comparison of model performance in summarization and Q&A tasks.
- Support varied content sources including PDFs and live web content.
- Offer model recommendations tailored to task type and content complexity.

---

## 🛠️ Tech Stack

| Component        | Technology               |
|------------------|--------------------------|
| Frontend UI      | Streamlit                |
| LLM Access       | Groq API                 |
| PDF Extraction   | PyMuPDF (`fitz`)         |
| Web Scraping     | BeautifulSoup, Requests  |
| Image Support    | PIL (optional)           |

---

## 🧠 Models Used

| Model Name               | Type            | Parameters | Best For                               |
|--------------------------|-----------------|------------|-----------------------------------------|
| LLaMA 3.3 70B Versatile  | Text Generation | 70B        | Deep summarization, contextual Q&A      |
| Gemma 2 9B Instruct      | Text Generation | 9B         | Instructional summaries, fast output    |
| LLaMA 3.1 8B Instant     | Text Generation | 8B         | Real-time summarization and chat        |
| LLaMA Guard 12B          | Moderation      | 12B        | Safety filtering and content screening  |
| Whisper Models (planned) | Speech-to-Text  | 1.5B       | Transcription (future support)          |

---

## 📷 Screenshots

*(Insert screenshots or gifs of the interface showing PDF upload, summary generation, and Q&A results.)*

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-document-analyzer.git
cd ai-document-analyzer
