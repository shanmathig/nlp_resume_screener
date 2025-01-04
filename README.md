# Resume Screening App

A machine learning-based application for automatically categorizing resumes into predefined job categories. The app processes resumes in PDF and TXT formats and provides a predicted category for the uploaded resume.

## Features:
- Upload resumes in PDF or TXT format.
- The app will extract the text and preprocess it.
- Predict the category of the resume using a pre-trained machine learning model.
- Display the predicted category on the web app.

## Technologies:
- **Streamlit**: For building the web interface.
- **scikit-learn**: For machine learning model and vectorization.
- **PyPDF2**: For extracting text from PDF files.
- **pickle**: For loading pre-trained machine learning models and encoders.

---

## Requirements

Before running the app, make sure to install the necessary Python packages:

```bash
pip install streamlit
pip install scikit-learn
pip install PyPDF2
