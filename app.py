import streamlit as st # for building 
import pickle
import nltk
import re
from resume_screener import cleanResume

nltk.download('punkt')
nltk.download('stopwords')

# loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# web app

def main():
    st.title('Resume Screening App')
    uploaded_file = st.file_uploader('Upload Resume', type=['pdf', 'txt'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        # utf decoding fails
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleanResume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            20: "DevOps Engineer",
            24: "Python Developer",
            12: "Web Designing",
            13: "HR",
            10: "Hadoop",
            18: "ETL Developer",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and Fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate"
        }

        category_name = category_mapping.get(prediction_id, "Unkown")
        st.write("Predicted Category: ", category_name)
    else:
        st.warning("No file uploaded.")
# python main
if __name__ == "__main__":
    main()


