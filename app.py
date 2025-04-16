import streamlit as st
import pickle
import re
import pandas as pd
import nltk
import plotly.express as px

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))


# Function to clean resumes
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text


# Load dataset
dataset_path = "UpdatedResumeDataSet11.csv"
df = pd.read_csv(dataset_path)

# Streamlit Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Analysis", "Recommendation"])

# Resume Analysis Page
if page == "Analysis":
    st.title("Resume Analysis Dashboard")

    # Display pie chart of categories
    st.write("## Pie Chart of Category Values")
    value_counts = df.iloc[:, 0].value_counts()
    fig = px.pie(values=value_counts.values, names=value_counts.index)
    st.plotly_chart(fig, use_container_width=True)

    # Display top 5 job categories
    st.write("## Top 5 Jobs")
    Category = df['Category'].value_counts().head(5)
    st.bar_chart(Category)

    # Check if 'Company' column exists before using it
    if 'company' in df.columns:
        st.write("## Top 5 Companies")
        company = df['company'].value_counts().head(5)
        st.bar_chart(company)
    else:
        st.warning("Company column not found in dataset.")

    # Display state distribution analysis
    if 'State Name' in df.columns:
        st.write("## State Distribution Analysis")
        state_distribution = df['State Name'].value_counts()
        fig = px.pie(values=state_distribution.values, names=state_distribution.index)
        st.plotly_chart(fig, use_container_width=True)

    if 'State Code' in df.columns:
        st.write("## State Map")
        fig = px.choropleth(df, locationmode='USA-states', locations='State Code', scope="usa", color='State Name')
        st.plotly_chart(fig, use_container_width=True)

# Resume Recommendation Page
elif page == "Recommendation":
    st.title("Resume Screening App")

    uploaded_resume = st.file_uploader("Upload Resume", type=['txt', 'pdf'])

    if uploaded_resume is not None:
        try:
            resume_bytes = uploaded_resume.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

        # Mapping category ID to category name
        category_mapping = {
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
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
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category:", category_name)

