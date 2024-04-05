import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import re

# Load the trained model
with open('D:/project/Movie success prediction/movie_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Function to preprocess user input
def preprocess_input(user_input):
    # Create DataFrame from user input
    user_df = pd.DataFrame([user_input])
    
    # Preprocess description text
    user_df['Description'] = user_df['Description'].apply(lambda x: x.lower())  # Convert to lowercase
    user_df['Description'] = user_df['Description'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove punctuation

    user_df['Actors'] = user_df['Actors'].apply(lambda x: x.lower())  # Convert to lowercase
    user_df['Actors'] = user_df['Actors'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove punctuation

    user_df['Description'] = user_df['Description'].apply(lambda x: x.lower())  # Convert to lowercase
    user_df['Description'] = user_df['Description'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove punctuation
    
    # Combine actors and director into a single string
    #user_df['Cast'] = user_df['Actors'] + ', ' + user_df['Director']
    
    # Drop 'Actors' and 'Director' columns
    #user_df.drop(['Actors', 'Director'], axis=1, inplace=True)
    
    # One-hot encode genre columns
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama',
              'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance',
              'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']
    for genre in genres:
        user_df[genre] = user_input.get(genre, 0)  # If the genre is not selected, set it to 0
    
    return user_df

# Function to make predictions
def make_prediction(model, user_input):
    user_preprocessed = preprocess_input(user_input)
    prediction = model.predict(user_preprocessed)
    return prediction

# Sidebar for user input
st.sidebar.header('Enter Movie Details')

# User input fields
description = st.sidebar.text_area('Description', 'Three girls are kidnapped by a man with a diagnosed 23 distinct personalities. They must try to escape before the apparent emergence of a frightful new 24th.')
actors = st.sidebar.text_input('Actors', 'James McAvoy, Anya Taylor-Joy, Haley Lu Richardson, Jessica Sula')
director = st.sidebar.text_input('Director', 'M. Night Shyamalan')
year = st.sidebar.number_input('Year', min_value=1900, max_value=2100, value=2016)
runtime = st.sidebar.number_input('Runtime (Minutes)', min_value=0, value=117)
rating = st.sidebar.number_input('Rating', min_value=0.0, max_value=10.0, value=7.3, step=0.1)
revenue = st.sidebar.number_input('Revenue (Millions)', min_value=0.0, value=138.12)
metascore = st.sidebar.number_input('Metascore', min_value=0.0, max_value=100.0)
votes_transformed = st.sidebar.number_input('Votes Transformed', min_value=0.0, value=500.0)

# Genre checkboxes
st.sidebar.header('Genre')
genres = {
    'Action': st.sidebar.checkbox('Action'),
    'Adventure': st.sidebar.checkbox('Adventure'),
    'Animation': st.sidebar.checkbox('Animation'),
    'Biography': st.sidebar.checkbox('Biography'),
    'Comedy': st.sidebar.checkbox('Comedy'),
    'Crime': st.sidebar.checkbox('Crime'),
    'Drama': st.sidebar.checkbox('Drama', value=True),  # Set Drama as default
    'Family': st.sidebar.checkbox('Family'),
    'Fantasy': st.sidebar.checkbox('Fantasy'),
    'History': st.sidebar.checkbox('History'),
    'Horror': st.sidebar.checkbox('Horror'),
    'Music': st.sidebar.checkbox('Music'),
    'Mystery': st.sidebar.checkbox('Mystery'),
    'Romance': st.sidebar.checkbox('Romance'),
    'Sci-Fi': st.sidebar.checkbox('Sci-Fi'),
    'Sport': st.sidebar.checkbox('Sport'),
    'Thriller': st.sidebar.checkbox('Thriller', value=True),  # Set Thriller as default
    'War': st.sidebar.checkbox('War'),
    'Western': st.sidebar.checkbox('Western')
}

# Combine genre checkboxes into a single list
selected_genres = [genre for genre, selected in genres.items() if selected]

# Preprocess user input and make prediction
user_input = {
    'Description': description,
    'Actors': actors,
    'Director': director,
    'Year': year,
    'Runtime (Minutes)': runtime,
    'Rating': rating,
    'Revenue (Millions)': revenue,
    'Metascore': metascore,
    'Votes_transformed': votes_transformed,
    **{genre: 1 for genre in selected_genres}  # Add selected genres as binary indicators
}

try:
    prediction = make_prediction(best_model, user_input)
    st.write(f"Predicted Success: {prediction}")
except Exception as e:
    st.error(f"An error occurred: {e}")
