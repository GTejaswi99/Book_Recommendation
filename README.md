# # 📚 Book Recommendation System

This is a Streamlit web app that recommends books based on the genre or keywords you input. Instead of simple matching, it uses **TF-IDF vectorization** and **cosine similarity** to find books with the most similar genres or themes.

---

## 🚀 Features

- Intelligent recommendations using **TF-IDF Vectorizer**
- Accepts flexible input (e.g., "romantic thriller", "sci-fi mystery")
- Interactive and user-friendly Streamlit UI
- Displays book cover, title, author, year, and publisher
- Books are presented in a 3-column grid layout

---

## 🧠 How It Works

1. TF-IDF is used to vectorize the `Book-Genre` column.
2. User input (genre/keywords) is also vectorized.
3. Cosine similarity measures closeness between the input and all book genres.
4. Top similar books are recommended.

---

## 💻 Installation & Running the App

### 1. Clone this repository or save the script

# Start the Streamlit app -  streamlit run app.py
