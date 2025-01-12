import streamlit as st
import pandas as pd
import google.generativeai as genai
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Set up Google API keys
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_CX = st.secrets["GOOGLE_SEARCH_ENGINE_ID"]

# Initialize Karma Points
if "karma_points" not in st.session_state:
    st.session_state["karma_points"] = 0

# Function to update karma points
def update_karma_points():
    st.session_state["karma_points"] += 1

# Function to interact with Google Search API
def google_search(query, filters={}):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    response = service.cse().list(q=query, cx=GOOGLE_CX, **filters).execute()
    results = response.get("items", [])
    search_results = []

    for result in results:
        search_results.append({
            "Title": result.get("title"),
            "URL": result.get("link"),
            "Snippet": result.get("snippet"),
        })
    return search_results

# Function to extract and summarize web content
def summarize_content(web_text):
    # Use AI summarization with multiple passes if needed
    model = genai.GenerativeModel('gemini-1.5-flash')
    initial_summary = model.generate_content(web_text).text
    # Further refine summary by extracting keywords
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([web_text])
    sorted_indices = np.argsort(X.sum(axis=0).A1)[::-1]
    important_keywords = [vectorizer.get_feature_names_out()[i] for i in sorted_indices[:5]]  # top 5 keywords
    refined_summary = f"Keywords: {', '.join(important_keywords)}\n\n{initial_summary}"
    return refined_summary

# Streamlit UI
st.title("Welcome to Karma Browser")
st.sidebar.header("Features")
action = st.sidebar.radio("Choose an Action", ["Search Web", "Use AI", "Both"])
export_csv = st.sidebar.checkbox("Export Results as CSV")

# Display karma points
st.sidebar.markdown(f"### Karma Points: {st.session_state['karma_points']}")

if action == "Search Web":
    st.header("Search the Web & Earn Karma Points")
    query = st.text_input("Enter your search query:")
    filters = {
        'dateRestrict': 'y[1]'  # Example: filter results from the last year
    }
    if st.button("Search"):
        update_karma_points()
        results = google_search(query, filters)
        if results:
            st.success(f"Found {len(results)} results.")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            if export_csv:
                csv = results_df.to_csv(index=False)
                st.download_button(label="Download Results as CSV", data=csv, file_name="search_results.csv", mime="text/csv")
        else:
            st.warning("No results found.")

elif action == "Use AI":
    st.header("Use Gemini AI for Summarization")
    input_text = st.text_area("Enter the text to summarize:")
    if st.button("Summarize"):
        update_karma_points()
        if input_text:
            try:
                summary = summarize_content(input_text)
                st.subheader("Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please provide text to summarize.")

elif action == "Both":
    st.header("Search the Web and Use Gemini AI for Summarization")
    query = st.text_input("Enter your search query:")
    filters = {
        'dateRestrict': 'y[1]'  # Filter for the last year, adjust as needed
    }
    if st.button("Search and Summarize"):
        update_karma_points()
        # Step 1: Search Web
        results = google_search(query, filters)
        if results:
            st.success(f"Found {len(results)} results.")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            # Step 2: Summarize Web Content
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

                        # Generate refined summary
                        ai_summary = summarize_content(web_text)
                        summaries.append({"URL": url, "Summary": ai_summary})
                        st.markdown(f"**URL:** {url}")
                        st.write(ai_summary)
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
