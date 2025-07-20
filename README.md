@ -0,0 +1,54 @@
# 🎓 MOOC Course Recommendation System

An intelligent web-based recommendation system that helps users discover online courses tailored to their interests, learning goals, and engagement history. Built with Python, Flask, Pandas, and scikit-learn, this system combines content-based filtering (TF-IDF), collaborative filtering, and a genetic algorithm for final course selection.

---

## 📌 Features

- 🔐 User Registration & Login
- 🎯 Personalized Recommendations based on:
  - User interests (topics, difficulty, rating)
  - Previous engagement
  - Other users’ course preferences (collaborative filtering)
  - Genetic algorithm optimization
- 📚 Add/Remove Courses to/from Dashboard
- 📊 User Feedback Logging for future training
- 📈 Accuracy measured using Precision@K

---

## 🧠 Recommendation Strategy

### 1. Content-Based Filtering (TF-IDF)
- Uses TfidfVectorizer to vectorize course descriptions
- Calculates similarity between courses and user preferences using cosine similarity

### 2. Collaborative Filtering
- Uses user-course interaction matrix
- Computes similarity scores between users or items (e.g., using Pearson correlation)

### 3. Genetic Algorithm Optimization
- Selects the best subset of courses based on a fitness function considering:
  - User's preferred topics
  - Rating thresholds
  - Difficulty level alignment
- Uses selection, crossover, and mutation to evolve recommendations over generations

---

## 🚀 Technologies Used

| Tech            | Purpose                           |
|-----------------|-----------------------------------|
| Python          | Core application logic            |
| Flask           | Web framework                     |
| Pandas          | Data handling                     |
| scikit-learn    | TF-IDF vectorizer, similarity     |
| Jinja2          | HTML templating                   |
| HTML/CSS/Bootstrap | Frontend UI                    |
| SQLite / CSV    | User and course data storage      |

---

