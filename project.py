import streamlit as st
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from io import BytesIO

# 🔹 Load BERT Model (Cached)
@st.cache_resource
def load_bert_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_bert_model()

# 🔹 Fast Text Cleaning
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)  # Remove mentions & hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation/numbers
    return text

# 🔹 **Sentiment Analysis Function**
def get_sentiment(text):
    cleaned_text = clean_text(text)
    result = sentiment_pipeline(cleaned_text)[0]
    
    sentiment_label = result['label']
    sentiment_score = round(result['score'], 4)

    # 🔹 Add Neutral Category
    if sentiment_score < 0.70:
        sentiment_label = "Neutral"
    elif sentiment_label == "POSITIVE":
        sentiment_label = "Positive"
    else:
        sentiment_label = "Negative"
    
    return sentiment_label, sentiment_score

# 🔹 Load or Process Data
@st.cache_data
def load_or_process_data():
    processed_file = "processed_feedback.csv"

    if os.path.exists(processed_file):
        return pd.read_csv(processed_file)

    df = pd.read_csv("customer_feedback.csv", low_memory=False)
    df['cleaned_feedback'] = df['feedback_text'].apply(clean_text)

    feedback_list = df['cleaned_feedback'].tolist()
    results = sentiment_pipeline(feedback_list, batch_size=16, truncation=True)

    df['sentiment_label'] = ["Neutral" if r['score'] < 0.70 else ("Positive" if r['label'] == "POSITIVE" else "Negative") for r in results]
    df['sentiment_score'] = [round(r['score'], 4) for r in results]

    df.to_csv(processed_file, index=False)
    return df

df = load_or_process_data()

# 🔹 Streamlit UI
st.title("📊 BERT-Based Customer Feedback Sentiment Analysis")

# 🔹 Sidebar Filters
product_filter = st.sidebar.selectbox("🔍 Select Product", ["All"] + list(df["product_name"].unique()))
location_filter = st.sidebar.selectbox("📍 Select Location", ["All"] + list(df["customer_location"].unique()))
sentiment_filter = st.sidebar.selectbox("😀 Select Sentiment", ["All", "Positive", "Neutral", "Negative"])

filtered_df = df.copy()
if product_filter != "All":
    filtered_df = filtered_df[filtered_df["product_name"] == product_filter]
if location_filter != "All":
    filtered_df = filtered_df[filtered_df["customer_location"] == location_filter]
if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df["sentiment_label"] == sentiment_filter]

# 🔹 Show Filtered Data
st.write(f"Showing {len(filtered_df)} results")
st.dataframe(filtered_df)

# 🔹 📈 **Overall Sentiment Distribution**
st.subheader("📊 Overall Sentiment Distribution")
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(data=df, x='sentiment_label', hue='sentiment_label', palette='coolwarm', ax=ax, order=["Positive", "Neutral", "Negative"], legend=False)
st.pyplot(fig)

# 🔹 📈 **Filtered Sentiment Distribution**
st.subheader("📊 Filtered Sentiment Distribution")
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.countplot(data=filtered_df, x='sentiment_label', hue='sentiment_label', palette='coolwarm', ax=ax2, order=["Positive", "Neutral", "Negative"], legend=False)
st.pyplot(fig2)

# 🔹 **Live Sentiment Analysis**
st.subheader("📝 Live Sentiment Analysis")
user_feedback = st.text_area("Enter customer feedback:")
if user_feedback:
    sentiment, score = get_sentiment(user_feedback)
    st.write(f"**Sentiment:** {sentiment} (Score: {score}) ✅")

# 🔹 Download Processed Data
st.subheader("📥 Download Processed Data")
def convert_df_to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output

csv_file = convert_df_to_csv(df)
st.download_button(label="Download CSV", data=csv_file, file_name="processed_feedback.csv", mime="text/csv")

st.success("✅ Model Successfully Integrated!")
