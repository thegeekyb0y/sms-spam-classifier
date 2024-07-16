# SMS Spam Detector
![image](https://github.com/user-attachments/assets/de4b8c5a-e1f8-43df-bf54-343c172d1480)

## Project Overview
A machine learning-based web application that classifies SMS messages as spam or not spam. This project demonstrates the full pipeline of a machine learning project, from data preprocessing to model deployment via streamlit.

Live demo: [SMS Spam Detector App](https://sms-spamdetector.streamlit.app/)

## Features
- Text input for SMS messages
- Real-time classification of messages as spam or not spam
- Simple and intuitive user interface

## Technologies Used
- Python
- Scikit-learn for handling machine learning models
- NLTK for natural language processing
- Streamlit for web application development
- Pickle for model serialization and vectorization

## Project Structure
- `app.py`: Main Streamlit application file
- `model.pkl`: Serialized machine learning model
- `vectorizer.pkl`: Serialized text vectorizer
- `requirements.txt`: List of Python dependencies
- `email_spam_classifier.ipynb`: Additional exploratory data analysis and model training and evaluating.

## How to Run Locally
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

## Machine Learning Pipeline

1. Data Preprocessing:
   - Removed special characters and numbers
   - Converted text to lowercase
   - Tokenized messages
   - Removed stop words
   - Applied stemming to reduce words to their root form

2. Exploratory Data Analysis:
- Cleaned data: removed unnecessary columns and duplicates
- Graphs and Heatmap: Learnt about trends and its corelations using graphs and heatmaps
- WordClouds: Used wordclouds to visualise which were the most common words
- Found class imbalance: more ham than spam
- Key insight: Spam messages typically longer than ham

3. Feature Engineering:
   - Used TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
   - Experimented with different max_features settings for TF-IDF

4. Model Selection:
   - Tested models:
     - SVC
     - KNeighbors
     - MultinomialNB
     - DecisionTree
     - LogisticRegression
     - RandomForest
     - AdaBoost
     - BaggingClassifier
     - ExtraTrees
     - GradientBoosting
     - XGBoost
   - Also experimented with ensemble methods:
     - VotingClassifier
     - StackingClassifier
   - Final model chosen: **MultinomialNB** 

5. Model Evaluation:
   - Used accuracy and precision as key metrics
   - Compared performance across different models and configurations
   - Used train-test split for evaluation (80% train, 20% test)

## Future Improvements
1. Hyperparameter tuning: Systematically tune hyperparameters for top-performing models

2. Advanced NLP techniques: Incorporate word embeddings or transformer-based models for potentially improved performance

3. Multilingual: Develop the ability to classify data from different languages.

## Contributing
Contributions to improve the project are welcome. Please feel free to submit a Pull Request.

---
Connect with Me: [All my socials](https://www.bento.me/adityatiwari)

Project Link: [GitHub Repository URL](https://github.com/thegeekyb0y/sms-spam-classifier)
