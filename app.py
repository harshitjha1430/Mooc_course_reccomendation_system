from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import ast
import numpy as np
import io
import base64
from genetic_algorithm import GeneticAlgorithm

# Machine learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'a_very_secret_key'  # Replace with a secure key

# File paths for datasets
COURSES_FILE_PATH = r'D:\Mooc_course_reccomendation_system\datasets\MOOC.ods'
USER_DATA_FILE_PATH = r'D:\Mooc_course_reccomendation_system\datasets\mooc_user_data.ods'

# Initialize the global variable for courses_df and user_data_df
courses_df = pd.DataFrame()
user_data_df = pd.DataFrame()

def load_datasets():
    global courses_df, user_data_df  # Ensure we modify the global variables

    # Load courses data
    if os.path.exists(COURSES_FILE_PATH):
        courses_df = pd.read_excel(COURSES_FILE_PATH, engine='odf')
    else:
        courses_df = pd.DataFrame()  # Initialize as empty if file doesn't exist

    # Load user data
    if os.path.exists(USER_DATA_FILE_PATH):
        user_data_df = pd.read_excel(USER_DATA_FILE_PATH, engine='odf')
    else:
        # Create an empty 'Added Courses' column if it doesn't exist
        user_data_df = pd.DataFrame(columns=[
            'User ID', 'Name', 'Email', 'Age', 'Country',
            'Occupation', 'Learning Goals', 'Preferred Topics',
            'Preferred Difficulty Level', 'Skills Acquired', 'Password', 'Added Courses'
        ])

# Load datasets on startup
load_datasets()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    global user_data_df
    if request.method == 'POST':
        try:
            user_id = request.form['user_id']
            if user_id in user_data_df['User ID'].values:
                flash("This User ID already exists. Please choose a different one.", "danger")
                return redirect(url_for('register'))

            password = request.form['password']
            confirm_password = request.form['confirm_password']
            if password != confirm_password:
                flash("Passwords do not match.", "danger")
                return redirect(url_for('register'))

            user_data = {
                'User ID': user_id,
                'Name': request.form['name'],
                'Email': request.form['email'],
                'Age': int(request.form['age']),
                'Country': request.form['country'],
                'Occupation': request.form['occupation'],
                'Learning Goals': request.form['learning_goals'],
                'Preferred Topics': request.form['preferred_topics'],
                'Preferred Difficulty Level': request.form['preferred_difficulty'],
                'Skills Acquired': request.form['skills_acquired'],
                'Password': password,
                'Added Courses': "[]",  # Initialize Added Courses as an empty list
            }

            user_data_df = pd.concat([user_data_df, pd.DataFrame([user_data])], ignore_index=True)
            user_data_df.to_excel(USER_DATA_FILE_PATH, index=False, engine='odf')

            flash("Registration successful! Your profile has been saved.", "success")
            return redirect(url_for('home'))
        except Exception as e:
            flash(f"Error during registration: {str(e)}", "danger")
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    login_error = None
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        password = request.form.get('password')

        if user_id and password:
            user_row = user_data_df[user_data_df['User ID'] == user_id]
            if not user_row.empty and user_row.iloc[0]['Password'] == password:
                session['user_id'] = user_id
                flash("Login successful!", "success")
                return redirect(url_for('dashboard'))  # Redirect to dashboard after successful login
            else:
                login_error = "Invalid User ID or Password. Please try again."
        else:
            login_error = "Please enter both User ID and Password."

    return render_template('login.html', login_error=login_error)

@app.route('/get_recommendation', methods=['GET', 'POST'])
def get_recommendation():
    global courses_df
    load_datasets()

    # Convert all 'all_skill' values to strings
    courses_df['all_skill'] = courses_df['all_skill'].astype(str)

    if request.method == 'POST':
        topic = request.form.get('topic').strip().lower()
        difficulty = request.form.get('difficulty').strip().lower()
        min_rating = request.form.get('min_rating')

        # Handle min_rating safely
        try:
            min_rating = float(min_rating) if min_rating else 0.0
        except ValueError:
            min_rating = 0.0

        # Fetch the user's added courses
        user_id = session.get('user_id')
        added_courses = []
        if user_id:
            user_row = user_data_df[user_data_df['User ID'] == user_id]
            if not user_row.empty:
                added_courses = ast.literal_eval(user_row.iloc[0]['Added Courses']) if isinstance(user_row.iloc[0]['Added Courses'], str) else []
        added_course_names = {course['Course Name'] for course in added_courses}

        # Filter courses based on topic, difficulty, and minimum rating
        def topic_match(skills):
            skills_list = [skill.strip().lower() for skill in str(skills).split(',')]
            return any(topic in skill for skill in skills_list)

        courses_df['Difficulty Level'] = courses_df['Difficulty Level'].fillna('').astype(str)
        courses_df['Course Rating'] = pd.to_numeric(courses_df['Course Rating'], errors='coerce')
        courses_df = courses_df.dropna(subset=['Course Rating'])

        filtered_courses = courses_df[
            courses_df['all_skill'].apply(topic_match) &
            (courses_df['Difficulty Level'].str.lower() == difficulty) &
            (courses_df['Course Rating'] >= min_rating) &
            ~courses_df['Course Name'].isin(added_course_names)  # Exclude added courses
        ]
        filtered_courses = filtered_courses.drop_duplicates(subset='Course Name')

        filtered_courses = filtered_courses.nlargest(20, 'Course Rating')  # Increase the number of candidates

        if filtered_courses.empty:
            flash("No courses found matching your criteria.", "warning")
            return render_template('recommendations.html')

        # Use Genetic Algorithm to select the best courses
        ga = GeneticAlgorithm(courses_df=filtered_courses, user_preferences=topic, num_recommendations=3)
        best_solution = ga.run()

        # Convert the best solution to course recommendations
        best_recommendations = filtered_courses.iloc[best_solution].to_dict(orient='records')

        return render_template('recommendation_results.html', recommendations=best_recommendations)

    return render_template('recommendations.html')


@app.route('/add_to_dashboard', methods=['POST'])
def add_to_dashboard():
    user_id = session.get('user_id')
    if not user_id:
        flash("You must log in to add courses to your dashboard.", "danger")
        return redirect(url_for('login'))

    course_name = request.form.get('course_name')
    course_rating = float(request.form.get('course_rating'))
    course_url = request.form.get('course_url')

    # Find the user's data in the dataframe
    user_row = user_data_df[user_data_df['User ID'] == user_id]
    if not user_row.empty:
        # Check if the 'Added Courses' column exists, if not, initialize it as an empty list
        added_courses = user_row.iloc[0]['Added Courses']
        if isinstance(added_courses, str):  # if it's a string, deserialize it
            added_courses = ast.literal_eval(added_courses)
        elif not isinstance(added_courses, list):  # if it's not a list, initialize it
            added_courses = []

        # Create the new course entry
        course_entry = {
            'Course Name': course_name,
            'Course Rating': course_rating,
            'Course URL': course_url
        }

        # Append the new course entry to the list
        added_courses.append(course_entry)

        # Update the dataset with the new list of courses
        user_data_df.loc[user_row.index, 'Added Courses'] = str(added_courses)
        user_data_df.to_excel(USER_DATA_FILE_PATH, index=False, engine='odf')

        flash(f"Course '{course_name}' has been added to your dashboard.", "success")
    else:
        flash("User data not found.", "danger")

    return redirect(url_for('dashboard'))


@app.route('/logout')
def logout():
    session.clear()  # Clears all session data
    return redirect(url_for('home'))  # Redirects to the home page (index.html)

@app.route('/dashboard')
def dashboard():
    user_id = session.get('user_id')
    if not user_id:
        flash("You must log in to access the dashboard.", "danger")
        return redirect(url_for('login'))

    # Find the user data
    user_data = user_data_df[user_data_df['User ID'] == user_id].iloc[0]

    # Get user name and email
    user_name = user_data['Name']
    user_email = user_data['Email']
    user_id= user_data['User ID']
    # Deserialize 'Added Courses' column (if stored as a string, convert it back to a list)
    added_courses = ast.literal_eval(user_data['Added Courses']) if isinstance(user_data['Added Courses'], str) else []

    return render_template('dashboard.html', user_name=user_name, user_email=user_email, added_courses=added_courses)
@app.route('/drop_course', methods=['POST'])
def drop_course():
    user_id = session.get('user_id')
    if not user_id:
        flash("You must log in to drop a course.", "danger")
        return redirect(url_for('login'))

    # Get the course name to drop
    course_name = request.form.get('course_name')

    # Locate the user's data
    user_row = user_data_df[user_data_df['User ID'] == user_id]
    if not user_row.empty:
        # Deserialize the 'Added Courses' field
        added_courses = ast.literal_eval(user_row.iloc[0]['Added Courses']) if isinstance(user_row.iloc[0]['Added Courses'], str) else []

        # Remove the course with the given name
        updated_courses = [course for course in added_courses if course['Course Name'] != course_name]

        # Update the user data and save it back to the dataset
        user_data_df.loc[user_row.index, 'Added Courses'] = str(updated_courses)
        user_data_df.to_excel(USER_DATA_FILE_PATH, index=False, engine='odf')

        flash(f"Course '{course_name}' has been removed from your dashboard.", "success")
    else:
        flash("User data not found.", "danger")

    return redirect(url_for('dashboard'))

@app.route('/suggestions', methods=['GET'])
def suggestions():
    if 'user_id' not in session:
        flash("Please log in to access suggestions.")
 
        return redirect(url_for('login'))

    user_id = session['user_id']

    # Fetch the user's added courses
    user_courses = get_user_courses(user_id)
    added_courses_set = set(user_courses)  

    if not user_courses:
        flash("You haven't added any courses yet.")
        return redirect(url_for('dashboard'))

    # Generate course suggestions based on similarities of the added courses
    course_suggestions = generate_course_suggestions(user_courses)

    # Organize suggestions by the course they are related to
    categorized_suggestions = {course_name: [] for course_name in user_courses}

    # Loop through the generated suggestions and categorize them
    for suggestion in course_suggestions:
        related_course_name = suggestion['Related To']  
        if suggestion['Course Name'] not in added_courses_set:  
            categorized_suggestions[related_course_name].append(suggestion)

    return render_template('suggestion.html', suggestions=categorized_suggestions)

def generate_course_suggestions(user_courses):
    """Generate course suggestions based on the user's added courses."""
    suggested_courses = []

    # Fetch the user's added courses for exclusion
    user_id = session.get('user_id')
    added_courses_set = set(get_user_courses(user_id)) if user_id else set()

    # Loop through the user's added courses to fetch their corresponding skills
    for course_name in user_courses:
        course_row = courses_df[courses_df['Course Name'] == course_name]
        if not course_row.empty:
            # Get the 'all_skill' field for the current course
            all_skills = course_row.iloc[0]['all_skill']

            # Now we find similar courses based on 'all_skill'
            similar_courses = find_similar_courses(all_skills, course_name)

            # Add the similar courses to the suggested list (avoid duplicates)
            for similar_course in similar_courses:
                if similar_course['Course Name'] not in added_courses_set:  # Exclude added courses
                    similar_course['Related To'] = course_name  # Add reference to the course it was suggested from
                    if similar_course['Course Name'] not in [course['Course Name'] for course in suggested_courses]:
                        suggested_courses.append(similar_course)

    return suggested_courses


def find_similar_courses(all_skills, course_name):
    global courses_df

    # Fetch the user's added courses for exclusion
    user_id = session.get('user_id')
    added_courses_set = set(get_user_courses(user_id)) if user_id else set()

    # Exclude courses with empty 'all_skill' or invalid 'Course Rating'
    filtered_courses_df = courses_df[
        courses_df['all_skill'].notna() & 
        (courses_df['all_skill'].str.strip() != "") &
        courses_df['Course Rating'].apply(lambda x: isinstance(x, (int, float)) or x.replace('.', '', 1).isdigit())
    ]

    # Vectorize the skills to calculate similarity
    vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b', stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(filtered_courses_df['all_skill'].fillna(''))

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    course_index = filtered_courses_df[filtered_courses_df['Course Name'] == course_name].index[0]
    similarity_scores = cosine_sim[course_index]
    sorted_indices = similarity_scores.argsort()[::-1]

    # Get the rating and difficulty level of the current course
    current_course = filtered_courses_df.loc[course_index]
    current_difficulty = current_course['Difficulty Level'].strip().lower()
    current_rating = current_course['Course Rating']

    # Find similar courses while considering difficulty level and rating
    similar_courses = []
    for idx in sorted_indices:
        similar_course = filtered_courses_df.iloc[idx]
        if (similar_course['Course Name'] not in added_courses_set and
            similar_course['Course Name'] != course_name):
            # Match difficulty level and ensure rating is within ±0.5 range
            try:
                similar_course_rating = float(similar_course['Course Rating'])
                if (similar_course['Difficulty Level'].strip().lower() == current_difficulty and
                    abs(similar_course_rating - current_rating) <= 0.5):
                    similar_courses.append({
                        'Course Name': similar_course['Course Name'],
                        'Course Rating': similar_course['Course Rating'],
                        'Course URL': similar_course['Course URL']
                    })
            except ValueError:
                # Skip courses with invalid ratings (like "not calibrated")
                continue

        if len(similar_courses) >= 7:
            break

    return similar_courses
def get_user_courses(user_id):
    """Fetch user courses from the global user_data_df DataFrame."""
    user_data = user_data_df[user_data_df['User ID'] == user_id].iloc[0]
    added_courses = ast.literal_eval(user_data['Added Courses']) if isinstance(user_data['Added Courses'], str) else []

    # Return a list of course names from the added courses
    return [course['Course Name'] for course in added_courses]




if __name__ == '__main__':
    app.run(debug=True)