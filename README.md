# Job Posting Classification & Recommendation System

An end-to-end **Machine Learning + NLP project** that scrapes real-time job postings, clusters similar jobs using unsupervised learning, and recommends relevant jobs based on user skills through an interactive **Streamlit web application**.

This project demonstrates **web scraping, natural language processing, machine learning clustering, and interactive dashboard development**, making it suitable for **Data Analyst / Data Scientist / ML Engineer portfolios**.

---

# Problem Statement

Job seekers often struggle to identify relevant job opportunities among thousands of postings across multiple platforms. Searching manually using keywords can lead to irrelevant results and inefficient job discovery.

This project builds an automated **job classification and recommendation system** that:

• Collects real job postings
• Extracts required skills using NLP
• Groups similar jobs using machine learning
• Recommends relevant jobs based on user skill input

This helps users quickly discover **best-fit job opportunities based on their technical skills**.

---

# Project Overview

The system automatically performs the following tasks:

1. Scrapes job postings from **Karkidi**
2. Extracts job information:

   * Job Title
   * Company
   * Location
   * Experience
   * Skills
   * Summary
3. Converts textual skill data into numerical features using **TF-IDF**
4. Applies **KMeans clustering** to group similar jobs
5. Allows users to input their skills
6. Predicts the closest job cluster
7. Displays recommended jobs
8. Allows users to **download results as CSV**

---

# Application Workflow

User enters skills
↓
Streamlit Application
↓
Job Scraping from Karkidi
↓
Text Preprocessing
↓
TF-IDF Feature Extraction
↓
KMeans Clustering
↓
Cluster Prediction
↓
Recommended Jobs Display

---

# Application Interface

The application consists of **three main stages**:

### 1️⃣ Scrape Jobs

Fetches real job postings from **Karkidi** using BeautifulSoup.

Displayed information includes:

* Job Title
* Company
* Location
* Experience
* Summary

---

### 2️⃣ Preprocess & Cluster

This step performs machine learning processing:

• Cleans job data
• Extracts skill features
• Applies **TF-IDF vectorization**
• Performs **KMeans clustering**
• Assigns cluster labels to each job

Each cluster represents a hidden job category such as:

* Data Science
* Machine Learning
* Data Engineering
* Business Intelligence
* Analytics

---

### 3️⃣ Recommend Jobs

Users enter their skills (comma-separated):

Example:

Python, SQL, Machine Learning

The system then:

• Converts skills into TF-IDF vectors
• Predicts the closest job cluster
• Displays matching job postings
• Allows downloading results as CSV

---

# Example Use Case

If a user enters:

Python, NLP, Deep Learning

The system will:

1. Convert the skills into TF-IDF features
2. Predict the closest cluster
3. Display jobs related to **AI / Machine Learning roles**

---

# Machine Learning Approach

## Why Unsupervised Learning?

Job postings do not come with predefined categories.

Therefore, **KMeans clustering** is used to automatically discover natural job groupings based on skill similarity.

---

# Text Vectorization (TF-IDF)

TF-IDF converts text data into numerical vectors by:

• Highlighting important skill words
• Reducing importance of common words
• Representing job skill sets mathematically

Example implementation:

```python
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Skills'])
```

---

# Clustering Algorithm

KMeans groups job postings based on skill similarity.

```python
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
```

Each cluster represents a **hidden job category**.

---

# Model Persistence

The trained model and vectorizer are saved for reuse.

```python
joblib.dump(kmeans, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
```

This avoids retraining the model every time the application runs.

---

# Project Structure

```
job-classification-project
│
├── app.py                # Streamlit web application
├── scraper.py            # Job scraper using BeautifulSoup
├── model_training.py     # TF-IDF + KMeans clustering
├── clustered_jobs.csv    # Clustered job dataset
├── model.pkl             # Saved clustering model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Project dependencies
└── README.md
```

---

# Technologies Used

Python
Pandas
BeautifulSoup
Requests
Scikit-learn
TF-IDF
KMeans Clustering
Joblib
Streamlit

---

# Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

Main libraries:

```
streamlit
pandas
beautifulsoup4
requests
scikit-learn
joblib
```

---

# How to Run the Project

### 1. Clone the Repository

```
git clone https://github.com/your-username/job-classification-project.git
cd job-classification-project
```

---

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

### 3. Run the Application

```
streamlit run app.py
```

The application will open automatically in your browser.

---

# Key Features

Real-time job scraping
Automatic job categorization
Unsupervised machine learning clustering
Skill-based job recommendation
Downloadable job results (CSV)
Interactive Streamlit interface

---

# Skills Demonstrated

Web scraping
Natural language processing
Machine learning clustering
Recommendation systems
Data preprocessing
Model persistence
Interactive dashboard development

---

# Future Improvements

Replace TF-IDF with **BERT embeddings**
Add **cosine similarity ranking** for better recommendations
Add **job analytics dashboard** (jobs per cluster)
Deploy application on **Streamlit Cloud**
Add **data visualization charts**
Support multiple job websites

---

# Use Cases

Job recommendation platforms
Skill-based job matching systems
Career analytics tools
NLP portfolio project
Machine learning academic project

---

# Author

Rashad
Data Analyst | Python | SQL | Power BI | Machine Learning

GitHub: https://github.com/RASHAD750

