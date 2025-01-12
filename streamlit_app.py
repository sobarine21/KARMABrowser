import streamlit as st
import google.generativeai as genai
import requests
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from langdetect import detect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure the Gemini API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Set up Google Search API keys
API_KEY = st.secrets["GOOGLE_API_KEY"]
CX = st.secrets["GOOGLE_SEARCH_ENGINE_ID"]

# Karma point initialization
if "karma_points" not in st.session_state:
    st.session_state.karma_points = 0

# Streamlit App UI
st.title("AI-Powered Web Search and Summarization Tool")
st.write("This app combines the power of Google Search and Gemini AI to help you search the web and generate AI-based responses.")

# Sidebar navigation
st.sidebar.header("Select Mode")
mode = st.sidebar.radio("Choose your action:", ["Web Search", "AI Response"])

# Karma Points display
st.sidebar.markdown(f"**Karma Points**: {st.session_state.karma_points}")

# AI Response Mode
if mode == "AI Response":
    st.subheader("Generate Response Using Gemini AI")
    
    # Input prompt for Gemini AI
    prompt = st.text_input("Enter your prompt:", "Best alternatives to JavaScript?")
    
    if st.button("Generate AI Response"):
        try:
            # Load and configure the model
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            st.write("AI Response:")
            st.write(response.text)
            
            # Award karma point for using Gemini AI
            st.session_state.karma_points += 1
        except Exception as e:
            st.error(f"Error: {e}")

# Web Search Mode
if mode == "Web Search":
    st.subheader("Search the Web")
    
    # Input query for Google Search
    search_query = st.text_input("Enter search query:")
    
    if st.button("Search"):
        if search_query:
            try:
                # Perform web search using Google Custom Search API
                service = build("customsearch", "v1", developerKey=API_KEY)
                response = service.cse().list(q=search_query, cx=CX).execute()
                search_results = response.get('items', [])
                
                # Placeholder for detected matches
                detected_matches = []

                # Process each search result
                for result in search_results:
                    url = result['link']
                    st.write(f"Analyzing: {url}...")

                    try:
                        content_response = requests.get(url, timeout=10)
                        if content_response.status_code == 200:
                            web_content = content_response.text
                            soup = BeautifulSoup(web_content, "html.parser")
                            paragraphs = soup.find_all("p")
                            web_text = " ".join([para.get_text() for para in paragraphs])
                            
                            # Calculate similarity between user query and web content
                            vectorizer = TfidfVectorizer().fit_transform([search_query, web_text])
                            similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
                            
                            if similarity > 0.5:  # Threshold for matches
                                detected_matches.append({
                                    "URL": url,
                                    "Similarity": round(similarity * 100, 2),
                                    "Summary": web_text[:200] + "..."  # Displaying summary of web content
                                })
                    except Exception as e:
                        st.error(f"Error processing URL {url}: {e}")

                # Display results
                if detected_matches:
                    st.success(f"Found {len(detected_matches)} potential matches!")
                    results_df = pd.DataFrame(detected_matches)
                    st.dataframe(results_df)
                    
                    # Award karma point for web search
                    st.session_state.karma_points += 1

            except Exception as e:
                st.error(f"Error in Google search: {e}")
        else:
            st.warning("Please enter a search query.")

# Final reminder for Karma Points
st.sidebar.markdown(f"**Total Karma Points**: {st.session_state.karma_points}")
