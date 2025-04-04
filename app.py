import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\tejaswi.gogikar\Downloads\book_recommendation.csv")
    df = df.dropna(subset=["Book-Title", "Book-Genre", "Image-URL-M", "Book-Author", "Year-Of-Publication", "Publisher"])
    return df

df = load_data()

# Vectorize Book-Genre column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Book-Genre'])

# Streamlit UI
st.title("📚 Book Recommendation System")

# Let user input a genre or keywords
selected_input = st.text_input("Enter a genre or related keywords (e.g., 'mystery, thriller')")

if st.button("Recommend"):
    if selected_input.strip() == "":
        st.warning("Please enter a genre or keyword to get recommendations.")
    else:
        # Transform user input to match TF-IDF space
        user_tfidf = tfidf.transform([selected_input])
        cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

        # Get top N similar books
        top_indices = cosine_sim.argsort()[-15:][::-1]  # Top 15 results

        if len(top_indices) == 0 or cosine_sim[top_indices[0]] == 0:
            st.write("No similar books found for this genre.")
        else:
            cols = st.columns(3)
            for idx, book_idx in enumerate(top_indices):
                row = df.iloc[book_idx]
                col = cols[idx % 3]
                with col:
                    st.image(row['Image-URL-M'], caption=row['Book-Title'], width=150)
                    st.write(f"### {row['Book-Title']}")
                    st.write(f"**Author:** {row['Book-Author']}")
                    st.write(f"**Year:** {row['Year-Of-Publication']}")
                    st.write(f"**Publisher:** {row['Publisher']}")
                    st.write("---")
