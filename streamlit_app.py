import streamlit as st
import google.generativeai as genai
import requests
import time
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import langid

# Download the vader_lexicon resource
nltk.download('vader_lexicon')

# ---- Helper Functions ----

def initialize_session():
    """Initializes session state variables."""
    if 'session_count' not in st.session_state:
        st.session_state.session_count = 0
    if 'block_time' not in st.session_state:
        st.session_state.block_time = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def check_session_limit():
    """Checks if the user has reached the session limit and manages block time."""
    if st.session_state.block_time:
        time_left = st.session_state.block_time - time.time()
        if time_left > 0:
            st.error(f"You have reached your session limit. Please try again in {int(time_left)} seconds.")
            st.write("Upgrade to Pro for unlimited content generation.")
            st.stop()
        else:
            st.session_state.block_time = None

    if st.session_state.session_count >= 5:
        st.session_state.block_time = time.time() + 15 * 60  # Block for 15 minutes
        st.error("You have reached the session limit. Please wait for 15 minutes or upgrade to Pro.")
        st.write("Upgrade to Pro for unlimited content generation.")
        st.stop()

def generate_content(prompt):
    """Generates content using Generative AI."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating content: {e}")
        raise

def search_web(query):
    """Searches the web using Google Custom Search API and returns results."""
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": st.secrets["GOOGLE_API_KEY"],
        "cx": st.secrets["GOOGLE_SEARCH_ENGINE_ID"],
        "q": query,
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        return response.json().get("items", [])
    else:
        st.error(f"Search API Error: {response.status_code} - {response.text}")
        return []

def display_search_results(search_results):
    """Displays search results in a structured format."""
    if search_results:
        st.warning("Similar content found on the web:")

        for result in search_results[:5]:  # Show top 5 results
            with st.expander(result['title']):
                st.write(f"**Source:** [{result['link']}]({result['link']})")
                st.write(f"**Snippet:** {result['snippet']}")
                st.write("---")

        st.warning("To ensure 100% originality, you can regenerate the content.")
        if st.button("Regenerate Content"):
            regenerate_and_display_content()
    else:
        st.success("No similar content found online. Your content seems original!")

def regenerate_and_display_content():
    """Regenerates content and displays it after ensuring originality."""
    original_text = st.session_state.generated_text
    regenerated_text = regenerate_content(original_text)
    st.session_state.generated_text = regenerated_text
    st.success("Content has been regenerated for originality.")
    st.subheader("Regenerated Content:")
    st.write(regenerated_text)

def regenerate_content(original_content):
    """Generates rewritten content based on the original content to ensure originality."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Rewrite the following content to make it original and distinct. Ensure it is paraphrased and does not match existing content:\n\n{original_content}"
    response = model.generate_content(prompt)
    return response.text.strip()

# Sentiment Analysis Functions

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

def analyze_vader_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

# Keyword Extraction

def extract_keywords(text):
    vectorizer = CountVectorizer(stop_words='english', max_features=10)  # Extract top 10 frequent words
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return keywords

# Language Detection

def detect_language(text):
    lang, confidence = langid.classify(text)
    return lang, confidence

# Word Frequency Analysis

def word_frequency(text):
    words = text.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(20)
    return most_common_words

# Word Cloud Generation

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# Text Summarization

def summarize_text(text, num_sentences=5):
    blob = TextBlob(text)
    sentences = blob.sentences
    scored_sentences = sorted(sentences, key=lambda s: s.sentiment.polarity, reverse=True)
    summary = ' '.join([str(sentence) for sentence in scored_sentences[:num_sentences]])
    return summary

# Chat Functionality

def chat_with_ai(prompt):
    """Chat with AI using Generative AI."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error during chat: {e}")
        raise

# Main Streamlit App

# Configure the API keys securely using Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# App Title and Description
st.title("AI-Powered Ghostwriter")
st.write("Generate high-quality content and check for originality using the power of Generative AI and Google Search.")

# Initialize session tracking
initialize_session()

# Prompt Input Field
prompt = st.text_area("Enter your prompt:", placeholder="Write a blog about AI trends in 2025.")

# Session management to check for block time and session limits
check_session_limit()

# Generate Content Button
if st.button("Generate Response"):
    if not prompt.strip():
        st.error("Please enter a valid prompt.")
    else:
        try:
            # Generate content using Generative AI
            generated_text = generate_content(prompt)

            # Increment session count
            st.session_state.session_count += 1

            # Display the generated content
            st.subheader("Generated Content:")
            st.write(generated_text)

            # Check for similar content online
            st.subheader("Searching for Similar Content Online:")
            search_results = search_web(generated_text)

            display_search_results(search_results)

        except Exception as e:
            st.error(f"Error generating content: {e}")

# Display regenerated content if available
if 'generated_text' in st.session_state:
    st.subheader("Regenerated Content (After Adjustments for Originality):")
    st.write(st.session_state.generated_text)

# Additional Analysis Tools

st.subheader("Additional Analysis Tools for Generated Content")

# Sentiment Analysis
if st.button("Analyze Sentiment"):
    if 'generated_text' in st.session_state:
        sentiment = analyze_sentiment(st.session_state.generated_text)
        st.write(f"Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")

        vader_sentiment = analyze_vader_sentiment(st.session_state.generated_text)
        st.write(f"Positive: {vader_sentiment['pos']}, Neutral: {vader_sentiment['neu']}, Negative: {vader_sentiment['neg']}")
    else:
        st.error("No generated content available for analysis.")

# Keyword Extraction
if st.button("Extract Keywords"):
    if 'generated_text' in st.session_state:
        keywords = extract_keywords(st.session_state.generated_text)
        st.write(keywords)
    else:
        st.error("No generated content available for keyword extraction.")

# Language Detection
if st.button("Detect Language"):
    if 'generated_text' in st.session_state:
        lang, confidence = detect_language(st.session_state.generated_text)
        st.write(f"Detected Language: {lang}, Confidence: {confidence}")
    else:
        st.error("No generated content available for language detection.")

# Word Frequency Analysis
if st.button("Analyze Word Frequency"):
    if 'generated_text' in st.session_state:
        word_freq = word_frequency(st.session_state.generated_text)
        st.write(word_freq)
    else:
        st.error("No generated content available for word frequency analysis.")

# Word Cloud Visualization
if st.button("Generate Word Cloud"):
    if 'generated_text' in st.session_state:
        wordcloud = generate_word_cloud(st.session_state.generated_text)
        st.image(wordcloud.to_array())
    else:
        st.error("No generated content available for word cloud generation.")

# Text Summarization
if st.button("Summarize Text"):
    if 'generated_text' in st.session_state:
        summary = summarize_text(st.session_state.generated_text)
        st.subheader("Text Summary")
        st.write(summary)
    else:
        st.error("No generated content available for summarization.")

# Chat with AI
st.subheader("Chat with AI")
chat_input = st.text_input("Enter your message for chat:")
if st.button("Send"):
    if chat_input:
        response = chat_with_ai(chat_input)
        st.session_state.chat_history.append({"user": chat_input, "ai": response})
    else:
        st.error("Please enter a message for chat.")

# Display Chat History
if st.session_state.chat_history:
    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**AI:** {chat['ai']}")
        st.write("---")
