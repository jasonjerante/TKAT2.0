import requests
import json
import pandas as pd
import unicodedata
import re
import os
import streamlit as st
import logging
from datetime import datetime, date
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base URL for downloading documents
BASE_URL = "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document({})/resource"

# File to store the last retrieval timestamp
TIMESTAMP_FILE = 'last_retrieval.txt'

# Set page configuration for Streamlit
st.set_page_config(layout="wide")


# Function to read the last retrieval date from the timestamp file
def get_last_retrieval_date():
    if os.path.exists(TIMESTAMP_FILE):
        with open(TIMESTAMP_FILE, 'r') as file:
            timestamp = file.read().strip()
            return datetime.fromisoformat(timestamp).date()
    return None


# Function to update the timestamp file with the current date
def update_last_retrieval_date():
    with open(TIMESTAMP_FILE, 'w') as file:
        file.write(datetime.now().isoformat())


# Step 1: Retrieve Data from OpenData Overheid Nederland API
def fetch_data(api_url):
    data = []
    while api_url:
        logging.info(f'Fetching data from API: {api_url}')
        response = requests.get(api_url)
        response_json = response.json()
        data.extend(response_json.get("value", []))
        api_url = response_json.get("@odata.nextLink")
    return data


def save_data_to_file(data, filename):
    logging.info(f'Saving data to file: {filename}')
    df = pd.DataFrame(data)
    df.to_csv(filename, sep='|', index=False)


def fetch_and_save_all_data():
    logging.info('Starting data retrieval from OpenData Overheid Nederland API.')
    document_url = "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document?$select=Onderwerp%2CVergaderjaar%2CDatumRegistratie%2CDatumOntvangst%2CAanhangselnummer%2CTitel%2CSoort%2CDocumentNummer%2CId&$filter=Verwijderd eq false and DatumRegistratie ge 2019-11-30T00:00:00Z&$orderby=DatumRegistratie asc"
    document_data = fetch_data(document_url)
    save_data_to_file(document_data, "results_Document.txt")
    update_last_retrieval_date()
    logging.info('Data retrieval completed.')


# Step 2: Clean and Transform the Data
def clean_data(filename):
    logging.info(f'Cleaning data from file: {filename}')
    df = pd.read_csv(filename, delimiter='|')

    for col in ['DatumRegistratie', 'DatumOntvangst']:
        df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
        df[col] = df[col].dt.date

    df.to_csv(f"output_{filename}", sep='|', index=False)
    logging.info(f'Data cleaned and saved to: output_{filename}')


def clean_special_characters(filename):
    logging.info(f'Cleaning special characters in file: {filename}')
    with open(filename, 'r') as infile:
        text = infile.read()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    with open(f"cleaned_{filename}", 'w') as outfile:
        outfile.write(text)
    logging.info(f'Special characters cleaned and saved to: cleaned_{filename}')


def transform_and_merge_data():
    logging.info('Starting data transformation and merging.')
    clean_data("results_Document.txt")
    clean_special_characters("output_results_Document.txt")

    document_df = pd.read_csv("cleaned_output_results_Document.txt", delimiter='|')
    # Assuming similar steps for DocumentActor and Persoon data
    # merged_df = pd.merge(document_df, documentactor_df, left_on="ID", right_on="Document_Id")
    # all_merged_df = pd.merge(merged_df, persoon_df, left_on="Persoon_Id", right_on="ID")

    document_df.to_csv("merged_all_data.txt", sep='|', index=False)
    logging.info('Data transformed and merged. Saved to merged_all_data.txt')


# Step 3: Visualize Data with Streamlit
def visualize_data():
    st.title('De Tweede Kamer Analyse Tool')
    st.write('Deze tool maakt alle openbare tweede kamer stukken overzichtelijk om analyses op los te laten.')

    last_retrieval_date = get_last_retrieval_date()
    today = date.today()

    if last_retrieval_date == today:
        st.write('Voor vragen of opmerkingen, neem contact met mij op via LinkedIn: https://www.linkedin.com/in/jkpstuve/. Laatste data update was op:', last_retrieval_date)
        logging.info('Data already up-to-date for today. Skipping data retrieval.')
    else:
        st.write('Data verwerken...')
        fetch_and_save_all_data()
        transform_and_merge_data()
        st.write('Data verwerken voltooid. Bekijk de resultaten:')

    df = pd.read_csv('merged_all_data.txt', delimiter='|')
    st.write("Data Overview")
    st.dataframe(df)  # Interactive table

    search_mode = st.radio("Kies de zoekmodus:", ('Een item', 'Meerdere items'))

    if search_mode == 'Een item':
        search_term = st.text_input('Vul hier uw zoekterm in:')
        if search_term:
            perform_search(df, [search_term.strip()], 'OR')
    else:
        search_term1 = st.text_input('Vul hier uw eerste zoekterm in:')
        search_term2 = st.text_input('Vul hier uw tweede zoekterm in:')
        search_logic = st.radio("Kies de logica:", ('AND', 'OR'))

        search_terms = []
        if search_term1:
            search_terms.append(search_term1.strip())
        if search_term2:
            search_terms.append(search_term2.strip())

        if search_terms:
            perform_search(df, search_terms, search_logic)


def perform_search(df, search_terms, search_logic):
    try:
        if search_logic == 'AND':
            search_results = df[df.apply(
                lambda row: all(term.lower() in ' '.join(row.astype(str).str.lower()) for term in search_terms),
                axis=1)]
        else:
            search_results = df[df.apply(
                lambda row: any(term.lower() in ' '.join(row.astype(str).str.lower()) for term in search_terms),
                axis=1)]

        st.write("Search Results")
        st.dataframe(search_results)  # Interactive table

        if not search_results.empty:
            # Time Trend per Month
            st.subheader('Time Trend per Month')
            search_results['DatumRegistratie'] = pd.to_datetime(search_results['DatumRegistratie'], errors='coerce')
            search_results['Month'] = search_results['DatumRegistratie'].dt.to_period('M')
            trend_data = search_results.groupby('Month').size().reset_index(name='Count')
            trend_data['Month'] = trend_data['Month'].dt.to_timestamp()

            fig = px.line(trend_data, x='Month', y='Count', title='Number of Documents per Month')
            st.plotly_chart(fig)

            # Linear Regression for Trend Analysis
            X = np.array(range(len(trend_data))).reshape(-1, 1)
            y = trend_data['Count'].values
            model = LinearRegression()
            model.fit(X, y)
            trend_line = model.predict(X)

            trend_data['Trend'] = trend_line
            fig = px.line(trend_data, x='Month', y=['Count', 'Trend'],
                          title='Number of Documents per Month with Trend Line')
            st.plotly_chart(fig)

            trend_slope = model.coef_[0]
            trend_direction = 'upward' if trend_slope > 0 else 'downward'
            st.write(f'The trend is {trend_direction} with a slope of {trend_slope:.2f} documents per month.')

            # Number of Documents Found
            st.subheader('Number of Documents Found')
            st.write(f'Total number of documents found: {len(search_results)}')

            # Breakdown of Document Types
            st.subheader('Breakdown of Document Types')
            doc_type_breakdown = search_results['Soort'].value_counts().reset_index()
            doc_type_breakdown.columns = ['Document Type', 'Count']
            st.write(doc_type_breakdown)

            fig = px.pie(doc_type_breakdown, values='Count', names='Document Type',
                         title='Breakdown of Document Types')
            st.plotly_chart(fig)

            # Document Download Links
            st.subheader('Document Download Links')
            for idx, row in search_results.iterrows():
                document_id = row['Id']
                document_subject = row['Onderwerp']
                download_link = BASE_URL.format(document_id)
                st.markdown(f"[{document_subject}]({download_link})", unsafe_allow_html=True)
        else:
            st.write("No search results found.")
    except Exception as e:
        logging.error("Error during search and display: %s", e)
        st.write("An error occurred while performing the search. Please adjust your search and try again.")


if __name__ == '__main__':
    logging.info('Starting application.')
    visualize_data()
    logging.info('Application running.')
