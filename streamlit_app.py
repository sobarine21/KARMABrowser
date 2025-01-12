import streamlit as st
import pandas as pd
import google.generativeai as genai
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from wordcloud import STOPWORDS

# Set up API keys securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Google API keys
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_CX = st.secrets["GOOGLE_SEARCH_ENGINE_ID"]

# Initialize Karma Points
if "karma_points" not in st.session_state:
    st.session_state["karma_points"] = 0

# Function to update karma points
def update_karma_points():
    st.session_state["karma_points"] += 1

# Function to interact with Google Search API
def google_search(query, content_type=""):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    if content_type:
        response = service.cse().list(q=query, cx=GOOGLE_CX, searchType=content_type).execute()
    else:
        response = service.cse().list(q=query, cx=GOOGLE_CX).execute()

    results = response.get("items", [])
    search_results = []
    
    for result in results:
        search_results.append({
            "Title": result.get("title"),
            "URL": result.get("link"),
            "Snippet": result.get("snippet"),
        })
    return search_results

# Keyword extraction function using TF-IDF
def extract_keywords(text):
    stop_words = set(stopwords.words('english')) | STOPWORDS
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5)
    tfidf_matrix = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return keywords

# Sentiment Analysis using TextBlob
def sentiment_analysis(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment

# Streamlit UI
st.title("Web Search & Gemini AI Summarization Tool")
st.sidebar.header("Advanced Features")

# Options for search or AI interaction
action = st.sidebar.radio("Choose an Action", ["Search Web", "Use AI", "Both"])
export_csv = st.sidebar.checkbox("Export Results as CSV")
karma_display = st.sidebar.markdown(f"### Karma Points: {st.session_state['karma_points']}")

# Content Type Filter for Google Search
content_filter = st.sidebar.selectbox("Select Content Type", ["All", "News", "Blogs", "Images", "Videos"])

# Display Karma Points
st.sidebar.markdown(f"### Karma Points: {st.session_state['karma_points']}")

# AI Interaction
if action == "Search Web":
    st.header("Search the Web")
    query = st.text_input("Enter your search query:")
    if st.button("Search"):
        update_karma_points()
        results = google_search(query, content_type=content_filter.lower())
        if results:
            st.success(f"Found {len(results)} results.")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            for idx, row in results_df.iterrows():
                st.markdown(f"**URL:** {row['URL']}")
                web_content = requests.get(row["URL"]).text
                st.markdown(f"**Sentiment:** {sentiment_analysis(web_content)}")
                st.markdown(f"**Keywords:** {', '.join(extract_keywords(web_content))}")
            if export_csv:
                csv = results_df.to_csv(index=False)
                st.download_button(label="Download Results as CSV", data=csv, file_name="search_results.csv", mime="text/csv")
        else:
            st.warning("No results found.")

elif action == "Use AI":
    st.header("Use Gemini AI for Summarization and Analysis")
    input_text = st.text_area("Enter the text to analyze:")
    if st.button("Analyze"):
        update_karma_points()
        if input_text:
            try:
                # Load and configure the model for summarization
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(input_text)
                st.subheader("AI Summary")
                st.write(response.text)

                # Sentiment Analysis
                sentiment = sentiment_analysis(input_text)
                st.subheader("Sentiment Analysis")
                st.write(f"Sentiment: {sentiment}")

                # Keyword Extraction
                keywords = extract_keywords(input_text)
                st.subheader("Extracted Keywords")
                st.write(f"Keywords: {', '.join(keywords)}")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please provide text to analyze.")

elif action == "Both":
    st.header("Search the Web and Use Gemini AI for Summarization")
    query = st.text_input("Enter your search query:")
    if st.button("Search and Analyze"):
        update_karma_points()
        # Step 1: Search Web
        results = google_search(query, content_type=content_filter.lower())
        if results:
            st.success(f"Found {len(results)} results.")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            # Step 2: Summarize Web Content with AI
            st.subheader("AI Summaries of Top Results")
            summaries = []
            for result in results[:3]:  # Limiting to top 3 results for summarization
                url = result["URL"]
                st.write(f"Analyzing: {url}")
                try:
                    content_response = requests.get(url, timeout=10)
                    if content_response.status_code == 200:
                        soup = BeautifulSoup(content_response.text, "html.parser")
                        paragraphs = soup.find_all("p")
                        web_text = " ".join([p.get_text() for p in paragraphs])

                        # Generate summary with Gemini AI
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        ai_summary = model.generate_content(web_text)
                        summaries.append({"URL": url, "Summary": ai_summary.text, "Sentiment": sentiment_analysis(web_text), "Keywords": ', '.join(extract_keywords(web_text))})

                        st.markdown(f"**URL:** {url}")
                        st.write(ai_summary.text)

                except Exception as e:
                    st.error(f"Error processing URL {url}: {e}")

            if export_csv and summaries:
                summaries_df = pd.DataFrame(summaries)
                csv = summaries_df.to_csv(index=False)
                st.download_button(label="Download Summaries as CSV", data=csv, file_name="summaries.csv", mime="text/csv")
        else:
            st.warning("No results found.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by Google Search API & Gemini AI")
