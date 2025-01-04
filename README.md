# Resume Category Prediction App

This project implements an NLP-based resume screening app that predicts job categories based on the text content of resumes, enabling HR professionals and hiring managers to quickly sort and categorize resumes. The app utilizes machine learning models, including a Support Vector Classifier (SVC), TF-IDF vectorization, and a label encoder for preprocessing.

## Features

- **Resume Upload:** Upload resumes in PDF or TXT format.
- **Text Extraction:** Extracts text from uploaded resumes.
- **Text Cleaning:** Cleans the extracted text by removing URLs, special characters, mentions, and more.
- **Prediction:** Predicts the job category for the resume using a pre-trained machine learning model.
- **User-Friendly Interface:** Built using Streamlit, providing an interactive and intuitive experience.

## Technologies Used

- **Streamlit:** For building the web interface.
- **Scikit-Learn:** For machine learning model training and evaluation.
- **TF-IDF Vectorization:** For transforming the text data into a suitable format for classification.
- **Pickle:** For saving and loading the pre-trained model, vectorizer, and label encoder.
- **PyPDF2:** For extracting text from PDF files.
- **Pandas:** For managing data and cleaning.
- **Matplotlib/Seaborn:** For visualizations (data distribution, category counts, etc.).

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/shanmathig/nlp_resume_screener.git
    cd nlp_resume_screener
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Download the pre-trained models:
    Make sure the following files are placed in the root directory:
    - `clf.pkl` (trained classifier)
    - `tfidf.pkl` (TF-IDF vectorizer)
    - `encoder.pkl` (label encoder)

    You can generate these files by running the provided Jupyter notebooks to train the models.

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
