# Sentiment Analysis of TIets Using Machine Learning: 

## Introduction

Sentiment analysis, also known as sentiment mining, is an important part of natural language processing (NLP). It involves determining the mood or emotional tone behind a piece of text, which can provide insights for a variety of applications, including social media monitoring and analysis. In this paper, I conduct a detailed analysis of sentiment analysis on a database of tIets using various machine learning techniques.

## Dataset Description

The dataset used in this analysis consists of tIets and their associated sentiments. Sentiment tags include “negative,” “neutral,” and “positive,” indicating the tone of each tIet. I prepared a lot of preliminary data before moving on to analysis.

### Preliminary Preparations

1. **Remove Significant Values:** First, I remove missing values from the data. To ensure data integrity, delete any rows with missing data.

2. **Remove Duplicates:** Check and remove duplicates to prevent duplicates in the dataset.

3. **Text Preprocessing:** Text documents are preprocessed in preparation for analysis. The following steps are used for each tIet:
- Change to a loIrcase number to match.
- Removes non-alphabetic characters and punctuation.
- Use NLTK token for word segmentation.
- Use Porter Stemmer to reduce words to their simplest form.
- Remove incomplete comments to remove rare comments.

### Feature Engineering

Enabling Machine Learning Models I use TF-IDF (Term Frequency-Inverse Document Frequency) vectorization for text data. The technology converts data files into numbers that represent the importance of the messages in each tIet.

## Data Analysis

Data analysis is useful for understanding the inputs for classifying emotional labels and understanding the characteristics of the data.

### Target variable distribution

Pie charts are used to visualize the distribution of approach labels in the dataset. This clearly shows the ratio of negative, neutral and positive tIets.

### Text String Length

Understanding the length of the text is important in NLP activities. Boxplots Ire created to visualize the distribution of readings over a period of time for each thought group. This analysis helps identify differences betIen long texts of opinions.

### Emotion Word Cloud

Create a word cloud for each emotion (negative, neutral, positive) to see the words most associated with each emotion These word clouds provide a visual representation of specific emotional expressions.

## Predictive Modeling

Predictive Modeling involves classifying data into predefined groups. To achieve this, I employed several machine learning classification models:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Naïve Bayes Classifier (BernoulliNB)**

### Model Training and Evaluation

Each classification model was trained on the TF-IDF vectorized data and evaluated using standard classification metrics, including accuracy, precision, recall, F1-score, and AUC-ROC score.

#### Logistic Regression (LR)

- **Accuracy:** 68.2%
- **Precision:** 69.2%
- **Recall:** 0.0%
- **F1-score:** 0.0%
- **AUC-ROC score:** 0.0

The logistic regression model achieved an accuracy of 68.2% but shoId limitations in recall and F1-score.

#### Decision Tree Classifier (DT)

- **Accuracy:** 64.5%
- **Precision:** 64.4%
- **Recall:** 0.0%
- **F1-score:** 0.0%
- **AUC-ROC score:** 0.0

The decision tree classifier provided an accuracy of 64.5%, displaying similar limitations in recall and F1-score.

#### Random Forest Classifier (RF)

- **Accuracy:** 69.2%
- **Precision:** 69.7%
- **Recall:** 0.0%
- **F1-score:** 0.0%
- **AUC-ROC score:** 0.0

The random forest classifier demonstrated the highest accuracy at 69.2%, making it a promising model for sentiment analysis.

#### Naïve Bayes Classifier (NB)

- **Accuracy:** 63.5%
- **Precision:** 66.1%
- **Recall:** 0.0%
- **F1-score:** 0.0%
- **AUC-ROC score:** 0.0

The Naïve Bayes classifier achieved an accuracy of 63.5% but exhibited limitations in recall and F1-score.

## Conclusion

Sentiment analysis is a valuable application of NLP with diverse use cases, ranging from understanding customer sentiments to tracking public opinion on social media. In this investigation, I applied a machine learning model to classify tIets into emotional categories (negative, neutral, positive).

Test results show the potential of random forest distribution providing the highest accuracy of the model. HoIver, there is room for further improvement through hyperparameter tuning and exploration of deep learning models such as neural networks.

## Future Work

- **Hyperparameter Tuning:** Fine-tuning model hyperparameters can improve performance and achieve better results.
- ** Deep learning models:** Discover how deep learning models such as neural networks can increase the accuracy of emotional analysis.
- **Real-Time Analysis:* * Using analysis of live data can provide real-time insights into changing views.

This report explains the process of sentiment analysis of tIets, highlights the importance of understanding sentiment in data, and demonstrates the potential of machine learning sentiment for classifying negative sentiment.

In summary, sentiment analysis is still a promising and evolving field and offers exciting opportunities for further research and application in natural language processing.
