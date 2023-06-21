#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis Project Todoist App Reviews on Google Playstore
# 
# _Author: Nanda Yuli Akhirdianto_
# 
# **Background:**
# In today's digital era, mobile applications are an important part of the daily lives of smartphone users. One of the more popular apps used to manage tasks and lists is Todoist. This application allows users to manage schedules, create to-do lists, set reminders, and much more. To understand the user's view of this application, sentiment analysis of user reviews on Google Playstore is important.
# 
# **Objective:**
# The aim of this project is to carry out a sentiment analysis of the Todoist app reviews received from users on the Google Playstore. Through this analysis, we can understand whether the views of users in general tend to be positive, negative, or neutral towards this application. In addition, we can identify specific aspects that users might highlight, both positive and negative, which can provide app developers with valuable insights in improving the user experience.
# 
# **Data:**
# The data used in this project comes from user reviews taken from Google Playstore. This data includes attributes such as review ID, username, review content, review score, time of review, version of the app when the review was created, and more. This data has been scraped from Twitter using a special scraping application. The data is then stored in a tabular format using the relevant columns for sentiment analysis.
# 
# By conducting a sentiment analysis of these Todoist app reviews, we can gain valuable insight into how users view the app as a whole. This information can assist application developers in identifying application strengths and weaknesses, as well as provide a solid basis for decision-making in improving the user experience.
# 
# #### **Setup**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('notebook/sentiment_app_review_analysis/reviews_app.csv')


# In[2]:


df = data[data['appId'] == 'com.todoist']
df.sample(10)


# #### **Data Validation and Cleaning Process**

# In[3]:


len(df)


# In[4]:


# Remove some columns that aren't used in further process
df = df.drop('reviewId', axis=1)
df = df.drop('userImage', axis=1)
df = df.drop('thumbsUpCount', axis=1)
df = df.drop('reviewCreatedVersion', axis=1)
df = df.drop('replyContent', axis=1)
df = df.drop('repliedAt', axis=1)
df = df.drop('sortOrder', axis=1)
df = df.drop('appId', axis=1)

print(df.columns)


# In[5]:


# Geting to know about data description
print(df.shape)
print(df.info())

# Getting to know how much rate that users used to give
print(df.loc[:, 'score'].mean())


# In[6]:


# Change some columns name for better understanding
df = df.rename(columns={
    'userName': 'user',
    'content': 'reviews',
    'score': 'rating',
    'at': 'date',
    'appVersion': 'app_ver'
})

print(df.columns)


# In[7]:


# format 'date' data type into datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
print(df.info())


# In[8]:


# Identify if any missing values and duplicate data
print(df.isnull().sum())
print(df.duplicated())


# In[9]:


# There are 176 rows that have missing value, since those missing value are on string we put 'unknown version' into it.
df['app_ver'].fillna('unknown version', inplace=True)

print(df.head())


# In[10]:


doist = df
doist.head()


# #### **Analyze Process**
# 
# We're conducting data exploration to interpret and gaining insight from this dataset by answering some questions.

# In[11]:


# Displays the distribution of scores (ratings)
rating_counts = doist['rating'].value_counts().sort_index()

# Make a bar plot for the distribution of scores (ratings)
plt.figure(figsize=(8, 6))
plt.bar(rating_counts.index, rating_counts.values)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of Ratings')
plt.show()


# That chart proves that the average of rating this app has earned is 3.0. Next, we want to know the distribution of the number of reviews by user, as well as evaluate the distribution of ratings from each user to the application.

# In[12]:


# Identify the user with the most number of reviews (user)
top_reviewers = doist['user'].value_counts().head(5)
most_reviews_user = top_reviewers.index[0]

print("User with the most number of reviews:")
print(most_reviews_user)

# Analyze trends or patterns in the most user reviews
user_reviews = doist[doist['user'] == most_reviews_user]
review_counts = user_reviews['app_ver'].value_counts().sort_index()

# Generates a bar plot for the most trending user reviews
plt.figure(figsize=(8, 6))
plt.bar(review_counts.index, review_counts.values)
plt.xlabel('Version')
plt.ylabel('Count')
plt.title('Review Trend for User: ' + most_reviews_user)
plt.xticks(rotation=90)
plt.show()


# Next, we want to see the relationship between rating and app version.

# In[13]:


# Calculates the average review score based on the app version and review time
version_ratings = doist.groupby(['app_ver'])['rating'].mean().reset_index()

# Sort data by AppCreatedVersion chronologically
version_ratings = version_ratings.sort_values(by='app_ver')

# Create line plots to see trends in changes in user ratings
plt.figure(figsize=(20, 6))
plt.plot(version_ratings['app_ver'], version_ratings['rating'], marker='o')
plt.xlabel('App Created Version')
plt.ylabel('Average Rating')
plt.title('Rating Trend by App Created Version')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# #### **Text Preprocessing**
# 
# Next process we want to analyze sentiment from all the reviews that users proceed in this app. This process we are conducting casefolding as part of the cleaning the text, then we're going to use package 'nltk' who provide modul sentiment.vader for easier step to labeling to find which one has positive, negative and neutral sentiment.

# In[14]:


import string
import re
import emoji

reviews = doist


# In[15]:


# use only reviews column for this process
reviews = reviews.drop('user', axis=1)
reviews = reviews.drop('rating', axis=1)
reviews = reviews.drop('date', axis=1)
reviews = reviews.drop('app_ver', axis=1)

reviews.head()


# In[16]:


# preprocess text and clean it
def preprocess_text(sentence):
    lower_case = sentence.lower()
    result = re.sub(r"\d+", "", lower_case)
    result = result.translate(str.maketrans("", "", string.punctuation))
    result = result.strip()
    result = emoji.demojize(result) # Mengubah emoji menjadi teks
    result = re.sub(r":\S+:", "", result) # Menghapus teks emoji
    return result

# put clean text into dataframe
reviews['clean_text'] = reviews['reviews'].apply(preprocess_text)
reviews.head()


# SentimentIntensityAnalyzer adalah kelas yang digunakan untuk melakukan analisis sentimen menggunakan metode VADER (Valence Aware Dictionary and sEntiment Reasoner). Metode ini menggunakan kamus yang sudah terlatih untuk memberikan skor sentimen pada teks berdasarkan kata-kata yang terkandung di dalamnya. SentimentIntensityAnalyzer akan memberikan skor sentimen berdasarkan komponen positif, negatif, dan netral, serta skor komposit yang mencerminkan sentimen keseluruhan dari teks tersebut.

# In[17]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer # import modul to execute the task

# Labeling initiation
sentiments = SentimentIntensityAnalyzer()
reviews["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in reviews["clean_text"]]
reviews["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in reviews["clean_text"]]
reviews["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in reviews["clean_text"]]
reviews["Compound"] = [sentiments.polarity_scores(i)["compound"] for i in reviews["clean_text"]]

reviews.head()


# In[18]:


# creates a 'Sentiment' column that shows sentiment ('positive', 'negative', or 'neutral')
# based on values in the 'Compound' column.
score = reviews['Compound'].values
sentiment = []

for i in score:
  if i >= 0.05:
    sentiment.append('positive')
  elif i <= -0.05:
    sentiment.append('negative')
  else:
    sentiment.append('neutral')

reviews['Sentiment'] = sentiment
reviews.head()


# In[19]:


# View the number of reviews by sentiment
sentiment_counts = reviews['Sentiment'].value_counts()

# Shows a visualization of the number of reviews by sentiment
plt.style.use('seaborn-pastel')

plt.figure(figsize=(8, 6))
colors = ['#66c2a5', '#fc8d62', '#8da0cb']
explode = (0.1, 0.1, 0.1)
plt.pie(sentiment_counts, labels=sentiment_counts.index, colors=colors, autopct='%1.1f%%',
        startangle=90, explode=explode, shadow=True)
plt.title('Todoist App Reviews Sentiment Distribution')
plt.axis('equal')
plt.show()


# In[20]:


# save dataframe into local storage
reviews.to_csv('sentiment_review_data.csv', index=False)


# Based on the visualization and sentiment analysis results, it was found that the majority of reviews for the Todoist application on the Play Store had positive sentiments (63.6%). The number of reviews with negative sentiment (27.7%) is also significant, while the number of reviews with neutral sentiment (8.7%) is comparatively less.
# 
# In conclusion, most of the users of the Todoist application on the Play Store give positive reviews. This can be an indication that the application has good performance and meets user needs in terms of task and activity management. However, it's still worth paying attention to negative reviews to understand any issues or flaws that may need fixing to improve the overall user experience.

# The next step is to do a word frequency analysis on the review content, the goal is to find out the words that appear most often. This information can assist in understanding which features users talk about the most and can provide insight into the positive or negative aspects of the app.

# In[21]:


# load some relevant libraries
import nltk
from nltk.tokenize import word_tokenize

df = reviews

df = df.drop('Positive', axis=1)
df = df.drop('Negative', axis=1)
df = df.drop('Neutral', axis=1)
df = df.drop('Compound', axis=1)
df = df.drop('Sentiment', axis=1)

df.head()


# #### **Tokenizing**

# In[22]:


# use the nltk library to split sentences into word lists.
#function takes a sentence as input and returns a list of tokens.
def tokenize_text(sentence):
    tokens = nltk.tokenize.word_tokenize(sentence)
    return tokens

df['token'] = df['clean_text'].apply(tokenize_text)
df.head()


# #### **Stopwords Removal (Filtering)

# In[23]:


from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

print(stopwords)


# In[24]:


def stopword_text(tokens):
    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords:
            cleaned_tokens.append(token)
    return cleaned_tokens

df['stop'] = df['token'].apply(stopword_text)
df.head()


# #### **Stemming**

# In[25]:


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


# In[26]:


def stemming_text(tokens):
    hasil = [stemmer.stem(token) for token in tokens]
    return hasil

df['stemmed'] = df['stop'].apply(stemming_text)
df.head()


# The following is to visualize some of the keywords that often appear for some of the keywords that have been defined

# In[27]:


from nltk.probability import FreqDist

all_tokens = [token for sublist in df['stop'] for token in sublist]
freq_dist = FreqDist(all_tokens)
print(freq_dist.most_common())


# In[28]:


freq_dist.plot(30, cumulative=False)
plt.show()


# In[29]:


# plot the stopwords into visualization, this time we are going to use wordcloud to do it
from wordcloud import WordCloud
from wordcloud import STOPWORDS

text = " ".join(i for i in df.clean_text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[30]:


# save dataframe into local storage
df.to_csv('freq_word_data.csv', index=False)


# #### **Conclusion**
# 
# 
# From the process and the findings obtained above, the following conclusions can be drawn:
# 
# 1. App Popularity: The frequent occurrence of the word "app" indicates that users often talk about the app itself. This shows that the Todoist application is quite popular among users and attracts their interest.
# 
# 2. Focus on Tasks and Lists: Words like "task" and "list" indicate that users use this app to organize tasks and create lists. This shows that the Todoist application effectively meets user needs in terms of organization and task management.
# 
# 3. Positive User Experience: The word "good" indicates that the majority of reviews give positive feedback to this app. This shows that users are satisfied with the features and functionality offered by Todoist.
# 
# 4. Time Management: The word "time" indicates that the user associates this application with time management and schedule management. This shows that Todoist helps users manage their time more efficiently.
# 
# 5. Widgets and Reminders: Words like "widgets" and "reminders" denote specific features in the Todoist application. These features provide added value to the user by providing easy access and reminders for important tasks.
# 
# Overall, these findings indicate that Todoist is a popular and effective application in helping users organize and manage their tasks. Features such as time management, reminders, and widgets provide a satisfying experience for users in using this application.
