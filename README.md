SMS Spam Detection

Overview

SMS Spam Detection is a machine learning-based project that classifies text messages as either Spam or Not Spam (Ham). The project utilizes Natural Language Processing (NLP) techniques to preprocess the text and train a model for accurate classification.


 Features
 
 Detects spam messages using machine learning
 
 Uses TF-IDF Vectorization for text processing
 
 Supports Naïve Bayes, Random Forest, and SVM models
 
 Interactive Streamlit Web App for real-time predictions
 
 Lightweight and easy to deploy

 Tech Stack

 Python – Programming Language
 
 Pandas, NumPy – Data Processing
 
 NLTK, Scikit-learn – Machine Learning & NLP
 
 Streamlit – Web App Interface
 
 Jupyter Notebook – Model Development


 Dataset

 We use the SMS Spam Collection Dataset available. The dataset contains ham (not spam) and spam messages, labeled accordingly.


 How It Works

 Data Preprocessing – Remove stopwords, punctuation, and tokenize text.
 
 Feature Extraction – Convert text into numerical form using TF-IDF.
 
 Model Training – Train Naïve Bayes / SVM / Random Forest model.
 
 Prediction – Classify messages as Spam or Ham.
 
 Deployment – Web-based UI using Streamlit.


🔮 Future Enhancements

✅ Improve model accuracy with deep learning (LSTM, BERT).
✅ Deploy as a web API for real-world applications.
✅ Add multilingual spam detection.
