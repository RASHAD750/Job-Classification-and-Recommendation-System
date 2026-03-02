#  Job Classification & Recommendation System

An end-to-end Machine Learning project that scrapes real-time job postings, performs unsupervised clustering using NLP techniques, and provides personalized job recommendations through an interactive Streamlit web application.

---

##  Project Overview

This project automatically:

1. Scrapes job postings from Karkidi
2. Extracts job title, company, skills, summary, and experience
3. Converts text data into numerical features using TF-IDF
4. Applies KMeans clustering to group similar jobs
5. Recommends jobs to users based on their skill input
6. Allows downloading matched jobs as CSV

This system demonstrates:

- Web Scraping
- Natural Language Processing (NLP)
- Unsupervised Machine Learning
- Model Persistence
- Full Deployment using Streamlit

---

##  Project Architecture

```
User Input (Skills)
        ↓
Streamlit Web App
        ↓
Job Scraper (BeautifulSoup)
        ↓
Preprocessing & TF-IDF Vectorization
        ↓
KMeans Clustering
        ↓
Skill-Based Cluster Prediction
        ↓
Recommended Jobs Output (Downloadable CSV)
```

---

##  Machine Learning Approach

###  Why Unsupervised Learning?

Job postings are not pre-labeled into categories.  
Therefore, KMeans clustering is used to automatically group similar jobs based on skill similarity.

###  Text Vectorization

TF-IDF (Term Frequency – Inverse Document Frequency) is used to:

- Convert textual skill data into numerical vectors
- Reduce importance of common words
- Highlight distinguishing skills

###  Clustering Algorithm

KMeans groups jobs into `n_clusters = 5` clusters based on skill similarity.

Each cluster represents a hidden job category such as:

- Data Science
- Machine Learning
- Data Engineering
- Business Intelligence
- Analytics

---

##  Project Structure

```
job-classification-project/
│
├── app.py                   # Streamlit Web Application
├── scraper.py               # Job Scraper (BeautifulSoup)
├── model_training.py        # TF-IDF + KMeans Clustering
├── clustered_jobs.csv       # Output clustered dataset
├── model.pkl                # Trained KMeans model
├── vectorizer.pkl           # Saved TF-IDF vectorizer
├── requirements.txt         # Dependencies
└── README.md                # Project Documentation
```

---

##  Technologies Used

- Python
- Pandas
- BeautifulSoup
- Requests
- Scikit-learn
- TF-IDF
- KMeans
- Joblib
- Streamlit

---

##  How to Run This Project

### 1️. Clone the Repository

```bash
git clone https://github.com/your-username/job-classification-project.git
cd job-classification-project
```

### 2️. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

##  Application Workflow

### Step 1: Scrape Jobs
Click **"Scrape Jobs"** to fetch job postings from Karkidi.

### Step 2: Preprocess & Cluster
Click **"Preprocess & Cluster"** to:
- Apply TF-IDF
- Train KMeans
- Assign cluster labels

### Step 3: Recommend Jobs
Enter your skills:

```
Python, SQL, Machine Learning
```

Click **"Recommend Jobs"**

The system:
- Converts your skills to TF-IDF
- Predicts cluster
- Displays matching jobs
- Allows CSV download

---

##  Sample Use Case

If a user enters:

```
Python, NLP, Deep Learning
```

The system will:
- Predict the closest job cluster
- Display jobs related to ML / AI roles
- Provide downloadable results

---

##  Model Design Details

### TF-IDF Vectorization

```python
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Skills'])
```

### KMeans Clustering

```python
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
```

### Model Persistence

```python
joblib.dump(kmeans, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
```

---

##  Key Features

✔ Real-time Job Scraping  
✔ Automatic Job Categorization  
✔ Unsupervised Machine Learning  
✔ Skill-Based Recommendation  
✔ Downloadable Results  
✔ Clean Interactive UI  

---

##  Future Improvements

- Replace TF-IDF with BERT embeddings
- Use cosine similarity ranking instead of cluster matching
- Add dashboard analytics (jobs per cluster visualization)
- Deploy to Streamlit Cloud
- Add keyword frequency visualization

---

##  Possible Interview Questions

### Why did you use KMeans?
Because job categories were not labeled. KMeans allows discovering natural groupings in the dataset.

### Why TF-IDF?
TF-IDF highlights important skill words while reducing noise from common terms.

### How can this be improved?
Using transformer embeddings (BERT) for better semantic understanding and ranking with cosine similarity.

---

