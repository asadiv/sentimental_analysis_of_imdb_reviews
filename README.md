# IMDb Sentiment Analysis using Machine Learning

A machine learning project that classifies IMDb movie reviews as **positive** or **negative** using Natural Language Processing and Logistic Regression.


## Live Demo
 [Add your Streamlit app link here]


---
## Out of core learning
The description that is given below is not about the outofcorelearning method, the outofcorelearning.py file is added just as a helpful material which is usefull in case of large data whcih can exceed memeory limit and continously flowing in data.

## Model Pipeline

1. Text Preprocessing
2. Tokenization / Stemming
3. TF-IDF Vectorization
4. Logistic Regression Classification


##  Data Preprocessing

Steps applied:

- Remove HTML tags
- Lowercasing text
- Extract emoticons
- Remove special characters
- Tokenization


## Model Training

We use:

- TF-IDF Vectorizer
- Logistic Regression
- GridSearchCV for tuning

Best parameters are selected automatically.


## Model Performance

### Accuracy
- Train Accuracy: 0.900
- Test Accuracy: 0.907

### Evaluation Metrics

- Precision: 0.8997444466286613
- Recall: 0.9154
- F1 Score: 0.9075047090314265

---

### Confusion Matrix
<img width="515" height="432" alt="Screenshot 2026-06-17 at 20-29-55 " src="https://github.com/user-attachments/assets/784f3046-79cb-4e1d-92f0-6cf7e518986f" />


### ROC Curve
<img width="567" height="455" alt="Screenshot 2026-06-17 at 20-33-24 " src="https://github.com/user-attachments/assets/4ab8027a-cc8b-46d2-97e0-e0781a61a929" />



