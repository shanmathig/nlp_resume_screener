#!/usr/bin/env python
# coding: utf-8

# Welcome! Let's create a resume screening app using Natural Language Processing!

# Understanding Our Dataset

# In[115]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# In[116]:


df = pd.read_csv('ResumeDataset.csv')
df.head()


# In[117]:


df.shape


# In[118]:


df['Category'].value_counts() # number of resumes per category


# In[119]:


plt.figure(figsize= (15,5))
sns.countplot(df['Category'])
plt.show()


# In[120]:


counts = df['Category'].value_counts()
labels = df['Category'].unique()
plt.figure(figsize= (15,10))
plt.pie(counts,labels=labels)


# Cleaning The Resume Data

# In[121]:


df['Resume'][0]


# Cleaning Data: URLs, hashtags, mentions, special letters, punctuation

# In[122]:


import re
def cleanResume(txt):
    #Removes URLs like http://example.com (with a trailing space), emails, and hashtags
    cleanTxt = re.sub(r"http\S+\s", "", txt) 
    cleanTxt = re.sub(r"@\S+", "", cleanTxt)
    cleanTxt = re.sub(r"#\S+", "", cleanTxt)
    # Removes all occurrences of RT or cc
    cleanTxt = re.sub(r"RT|cc","", cleanTxt)
    # Remove special characters
    special_chars = """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
    # Escape the special characters manually in the regular expression
    cleanTxt = re.sub(r'[%s]' % re.escape(special_chars), ' ', cleanTxt)
    # Remove non-ASCII chars
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    # Remove multiple consecutive spaces
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt)

    return cleanTxt


# In[123]:


cleanResume("my ### #shan&mathi site is http://helowrld and access it @gmail.com")


# In[124]:


# Create a new column 'Cleaned_Resume' with the cleaned version
df['Cleaned_Resume'] = df['Resume'].apply(lambda x: cleanResume(x))

# Compare the original and cleaned columns
print(df[['Resume', 'Cleaned_Resume']].head())


# Resume Words -> Categories!

# In[125]:


import sklearn
from sklearn.preprocessing import LabelEncoder



# In[126]:


le = LabelEncoder()
le.fit(df['Category'])
# transforms the categorical values in the Category column into numeric labels
df['Category'] = le.transform(df['Category'])
df.Category.unique()
# 6 data science, 12 HR, 0 Advocste...


# Vectorization

# In[127]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words= 'english')

tfidf.fit(df['Resume'])
requiredText = tfidf.transform(df['Resume'])


# In[128]:


df


# Splitting

# In[129]:


from sklearn.model_selection import train_test_split


# In[130]:


x_train, x_test, y_train, y_test = train_test_split(requiredText, df['Category'], test_size=0.2, random_state=42)


# In[131]:


x_train.shape # 80%


# In[132]:


x_test.shape # 20%


# Training The Model

# In[133]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(x_train, y_train)

ypred = clf.predict(x_test)
print(ypred)
print(accuracy_score(y_test,ypred))



# Now that our model works with accuracy, let's create the prediction system!

# In[134]:


# Save data into 3 new files to reload later without having to vectorize again
import pickle

pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))


# In[135]:


myresume = """
Jane Smith is a certified personal trainer with over 5 years of experience in helping individuals achieve their fitness goals. Specializing in weight loss, strength training, and sports conditioning, Jane has developed personalized workout routines for clients of all ages and fitness levels. She has extensive knowledge in nutrition and exercise science, and uses this to create holistic health and fitness programs that are tailored to individual needs.

Jane holds a degree in Exercise Science and is a certified trainer through the National Academy of Sports Medicine (NASM). She has worked with athletes, seniors, and individuals with chronic health conditions, helping them improve their physical well-being and overall quality of life.

Her expertise includes:
- Weight Loss and Body Composition
- Strength Training and Resistance Exercises
- Cardio Conditioning
- Nutrition Coaching and Meal Planning
- Injury Prevention and Rehabilitation
- Functional Movement and Flexibility Training
- Group Fitness Classes

Certifications:
- Certified Personal Trainer, NASM
- CPR and First Aid Certified
- Yoga Instructor (200-Hour Certification)

Education:
BSc in Exercise Science, ABC University, 2014-2018

Work Experience:
- Personal Trainer at XYZ Fitness Gym (2018-Present)
- Fitness Coach at Wellness Center (2016-2018)

Languages:
- English (Fluent)
- Spanish (Conversational)
"""


# In[136]:


import pickle
clf = pickle.load(open('clf.pkl', "rb"))
cleaned_resume = cleanResume(myresume)

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
print("Predicted Category: ", category_name)
print(prediction_id)


# 
