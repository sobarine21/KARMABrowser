import streamlit as st
import pandas as pd
import google.generativeai as genai
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
from io import StringIO
from fpdf import FPDF

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
def google_search(query):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
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

# Function to generate a PDF from summaries
def generate_pdf(summaries_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt="AI Summarized Web Results", ln=True, align='C')

    # Add each summary as a new line in the PDF
    for index, row in summaries_df.iterrows():
        pdf.ln(10)  # Line break
        pdf.cell(200, 10, txt=f"URL: {row['URL']}", ln=True)
        pdf.multi_cell(0, 10, txt=f"Summary: {row['Summary']}")
    
    return pdf.output(dest='S').encode('latin1')

# Streamlit UI
st.title("KARMA The AI powered Browser")
st.sidebar.header("Features")
action = st.sidebar.radio("Choose an Action", ["Search Web", "Use AI", "Both"])
export_csv = st.sidebar.checkbox("Export Results as CSV")
export_txt = st.sidebar.checkbox("Export Results as TXT")
export_pdf = st.sidebar.checkbox("Export Results as PDF")

# Display karma points
st.sidebar.markdown(f"### Karma Points: {st.session_state['karma_points']}")

if action == "Search Web":
    st.header("Search the Web & Earn Karma Points")
    query = st.text_input("Enter your search query:")
    if st.button("Search"):
        update_karma_points()
        results = google_search(query)
        if results:
            st.success(f"Found {len(results)} results.")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            if export_csv:
                csv = results_df.to_csv(index=False)
                st.download_button(label="Download Results as CSV", data=csv, file_name="search_results.csv", mime="text/csv")
            if export_txt:
                txt = StringIO()
                results_df.to_string(txt, index=False)
                st.download_button(label="Download Results as TXT", data=txt.getvalue(), file_name="search_results.txt", mime="text/plain")
            if export_pdf:
                pdf = generate_pdf(results_df)
                st.download_button(label="Download Results as PDF", data=pdf, file_name="search_results.pdf", mime="application/pdf")
        else:
            st.warning("No results found.")

elif action == "Use AI":
    st.header("Use Gemini AI for Summarization")
    input_text = st.text_area("Enter the text to summarize:")
    if st.button("Summarize"):
        update_karma_points()
        if input_text:
            try:
                # Load and configure the model for summarization
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(input_text)
                st.subheader("Summary")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please provide text to summarize.")

elif action == "Both":
    st.header("Search the Web & Earn Karma points")
    query = st.text_input("Enter your search query:")
    if st.button("Search and Summarize"):
        update_karma_points()
        # Step 1: Search Web
        results = google_search(query)
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

                        # Generate summary with Gemini AI
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        ai_summary = model.generate_content(web_text)
                        summaries.append({"URL": url, "Summary": ai_summary.text})
                        st.markdown(f"**URL:** {url}")
                        st.write(ai_summary.text)
                except Exception as e:
                    st.error(f"Error processing URL {url}: {e}")

            # Check if summaries were successfully generated before proceeding with export
            if summaries:
                summaries_df = pd.DataFrame(summaries)

                # Export options
                if export_csv:
                    csv = summaries_df.to_csv(index=False)
                    st.download_button(label="Download Summaries as CSV", data=csv, file_name="summaries.csv", mime="text/csv")
                if export_txt:
                    txt = StringIO()
                    summaries_df.to_string(txt, index=False)
                    st.download_button(label="Download Summaries as TXT", data=txt.getvalue(), file_name="summaries.txt", mime="text/plain")
                if export_pdf:
                    pdf = generate_pdf(summaries_df)
                    st.download_button(label="Download Summaries as PDF", data=pdf, file_name="summaries.pdf", mime="application/pdf")
            else:
                st.warning("No summaries available to export.")
        else:
            st.warning("No results found.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by Google Search API & Gemini AI")
