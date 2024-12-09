import tkinter as tk
import customtkinter as ctk
import pandas as pd
import requests
import numpy as np
import webbrowser
from io import BytesIO
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
import pickle
from sklearn.preprocessing import LabelEncoder
from surprise import Dataset, SVD

### Model and Dataset Loading ############################################################

# Load CSV for names
df = pd.read_csv('../data/games.csv')
recommendations_df = pd.read_csv('../data/recommendations.csv')
game_names = df['name'].tolist()  # Convert the 'name' column to a list
class SVDWithProgress(SVD):
    def fit(self, trainset):
        print(f"Training SVD with {self.n_epochs} epochs...")
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}...")
            super().fit(trainset)
        return self

svd_model_path = 'svd_model.pkl'
game_encoder_path = 'game_encoder.pkl'

# Load the trained SVD model
try:
    with open(svd_model_path, 'rb') as model_file:
        model_svd = pickle.load(model_file)
except FileNotFoundError:
    print("SVD model file not found. Please train and save the model before running.")
    model_svd = None

# Load the trained game encoder
try:
    with open(game_encoder_path, 'rb') as encoder_file:
        game_encoder = pickle.load(encoder_file)
except FileNotFoundError:
    # If the encoder is not found, initialize a new LabelEncoder and fit it to your game names
    print("Game encoder not found, initializing a new one...")
    game_encoder = LabelEncoder()
    game_encoder.fit(df['name'])  # Assuming df is your game DataFrame with a 'name' column

    # Save the game encoder for future use
    with open(game_encoder_path, 'wb') as encoder_file:
        pickle.dump(game_encoder, encoder_file)

user_age = 50
user_budget = 250
user_language = "English"

###### CustomTkinter setup ################################################################
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

app = ctk.CTk()
app.geometry("800x600")
app.title("Game Recommender")

added_games = []
loaded_games = 0


def get_image(url):
    try:
        response = requests.get(url)
        image_data = response.content
        pil_image = Image.open(BytesIO(image_data))
        resized_image = pil_image.resize((240, 135))
        return ImageTk.PhotoImage(resized_image)
    except Exception as e:
        print(f"Error fetching image from {url}: {e}")
        return None

# Helper function for updating labels
def update_age_label(value):
    global user_age
    user_age = int(value) 
    age_label.configure(text=f"Age: {user_age}")

def update_budget_label(value):
    global user_budget
    user_budget = int(value)
    budget_label.configure(text=f"Budget: ${user_budget}")

# Screen 1: Age input
def age_screen():
    for widget in app.winfo_children():
        widget.destroy()

    frame = ctk.CTkFrame(app)
    frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    title_label = ctk.CTkLabel(frame, text="Enter Your Age", font=("AnonymicePro Nerd Font", 48))  # Double the font size
    title_label.pack(pady=40)

    age_slider = ctk.CTkSlider(frame, from_=0, to=100, command=update_age_label, number_of_steps=100)
    age_slider.set(user_age)  # Set to the global default age
    age_slider.pack(pady=20)

    global age_label
    age_label = ctk.CTkLabel(frame, text=f"Age: {user_age}", font=("AnonymicePro Nerd Font", 60))  # Larger font for age display
    age_label.pack(pady=40)

    next_button = ctk.CTkButton(frame, text="Next", command=budget_screen, font=("AnonymicePro Nerd Font", 30))  # Larger button font
    next_button.pack(pady=40)

# Screen 2: Budget input
def budget_screen():
    for widget in app.winfo_children():
        widget.destroy()

    frame = ctk.CTkFrame(app)
    frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    title_label = ctk.CTkLabel(frame, text="Enter Your Budget", font=("AnonymicePro Nerd Font", 48))  # Double the font size
    title_label.pack(pady=40)

    budget_slider = ctk.CTkSlider(frame, from_=0, to=100, command=update_budget_label, number_of_steps=100)
    budget_slider.set(user_budget)  # Set to the global default budget
    budget_slider.pack(pady=20)

    global budget_label
    budget_label = ctk.CTkLabel(frame, text=f"Budget: ${user_budget}", font=("AnonymicePro Nerd Font", 60))  # Larger font for budget display
    budget_label.pack(pady=40)

    next_button = ctk.CTkButton(frame, text="Next", command=language_screen, font=("AnonymicePro Nerd Font", 30))  # Larger button font
    next_button.pack(pady=40)

# Screen 3: Language input
def language_screen():
    global user_language
    for widget in app.winfo_children():
        widget.destroy()

    frame = ctk.CTkFrame(app)
    frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    title_label = ctk.CTkLabel(frame, text="Select Your Language", font=("AnonymicePro Nerd Font", 60), wraplength=600)  # Adjust wraplength as needed
    title_label.pack(pady=40)

    languages = ['English', 'Korean', 'Simplified Chinese', 'French', 'German', 'Spanish - Spain', 'Arabic', 'Japanese', 'Polish', 'Portuguese', 'Russian', 'Turkish', 'Thai', 'Italian', 'Portuguese - Brazil', 'Traditional Chinese', 'Ukrainian']
    language_var = tk.StringVar(value=user_language)
    language_dropdown = ctk.CTkOptionMenu(frame, variable=language_var, values=languages, font=("AnonymicePro Nerd Font", 20), width=400)  # Adjust font and width
    language_dropdown.pack(pady=40)

    def save_language():
        global user_language
        user_language = language_var.get()
        game_input_screen()

    next_button = ctk.CTkButton(frame, text="Next", command=save_language, font=("AnonymicePro Nerd Font", 30))  # Larger button font
    next_button.pack(pady=40)

# Screen 4: Game input with search and hours played
def game_input_screen():
    for widget in app.winfo_children():
        widget.destroy()

    left_frame = ctk.CTkFrame(app)
    left_frame.place(relx=0.3, rely=0.5, anchor=tk.CENTER)

    title_label = ctk.CTkLabel(left_frame, text="Add The Games You Have Played", font=("AnonymicePro Nerd Font", 24))
    title_label.pack(pady=20)

    # Search bar
    search_var = tk.StringVar()
    search_entry = ctk.CTkEntry(left_frame, textvariable=search_var)
    search_entry.pack(pady=10)

    # Display 10 game names at a time
    listbox_frame = ctk.CTkFrame(left_frame)
    listbox_frame.pack(pady=10)

    listbox = tk.Listbox(listbox_frame, height=10, width=50)
    listbox.pack(side=tk.LEFT, padx=5)

    scrollbar = tk.Scrollbar(listbox_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)

    def update_listbox(*args):
        search_term = search_var.get().lower()
        matching_games = [game for game in game_names if search_term in game.lower()][:10]  # Show top 10 matches
        listbox.delete(0, tk.END)
        for game in matching_games:
            listbox.insert(tk.END, game)

    search_var.trace_add("write", update_listbox)

    hours_played_label = ctk.CTkLabel(left_frame, text="Enter Hours Played", font=("AnonymicePro Nerd Font", 24))
    hours_played_label.pack(pady=10)

    hours_var = tk.StringVar()
    hours_entry = ctk.CTkEntry(left_frame, textvariable=hours_var)
    hours_entry.pack(pady=10)

    right_frame = ctk.CTkFrame(app)
    right_frame.place(relx=0.75, rely=0.5, anchor=tk.CENTER)

    play_history_label = ctk.CTkLabel(right_frame, text="Playtime History", font=("AnonymicePro Nerd Font", 24))
    play_history_label.pack(pady=20)

    history_frame = ctk.CTkFrame(right_frame)
    history_frame.pack()

    # Add the selected game and hours to the added games list and display
    def add_game():
        selected_game = listbox.get(tk.ACTIVE)
        hours = hours_var.get()
        if selected_game and hours:
            added_games.append((selected_game, hours))
            game_label = ctk.CTkLabel(history_frame, text=f"{selected_game} - {hours} hours", font=("AnonymicePro Nerd Font", 16), wraplength=200)
            game_label.pack(anchor="w", pady=5)

    add_button = ctk.CTkButton(left_frame, text="Add Game", command=add_game)
    add_button.pack(pady=10)

    next_button = ctk.CTkButton(left_frame, text="Finish", command=show_loading_screen)
    next_button.pack(pady=20)

# Show the loading screen with progress bar
def show_loading_screen():
    for widget in app.winfo_children():
        widget.destroy()

    frame = ctk.CTkFrame(app)
    frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    loading_label = ctk.CTkLabel(frame, text="A game recommendation is loading for you...", font=("AnonymicePro Nerd Font", 18))
    loading_label.pack(pady=20)

    status_label = ctk.CTkLabel(frame, text="Filtering games by age...", font=("AnonymicePro Nerd Font", 16))
    status_label.pack(pady=5)

    progress = ctk.CTkProgressBar(frame, width=300)
    progress.pack(pady=30)
    progress.set(0)

    def update_progress(value=0):
        progress.set(value)
        if value < 0.33:
            status_label.configure(text="Filtering games by age...")
        elif value < 0.66:
            status_label.configure(text="Filtering games by budget...")
        elif value < 1:
            status_label.configure(text="Filtering games by supported languages...")

        if value < 1:
            app.after(100, update_progress, value + 0.1)
        else:
            filtered_games = recommender(user_age, user_budget, [user_language], added_games, model_svd, game_encoder)
            show_recommendation_screen(filtered_games)

    update_progress()

# Show the game recommendation screen after progress bar completes
def show_recommendation_screen(recommended_games):
    global loaded_games
    loaded_games = 0

    for widget in app.winfo_children():
        widget.destroy()

    canvas = tk.Canvas(app, bg="#2a2d2e", highlightthickness=0)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(app, orient=tk.VERTICAL, command=canvas.yview, bg="#2a2d2e", highlightthickness=0)
    scrollbar.pack(side=tk.LEFT, fill=tk.Y)

    scrollable_frame = ctk.CTkFrame(canvas, fg_color="#2a2d2e", corner_radius=0)
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    see_more_button = ctk.CTkButton(scrollable_frame, text="See More ⬇️", font=("AnonymicePro Nerd Font", 20), command=lambda: load_more_games(see_more_button))

    # Function to load the next set of games
    def load_more_games(button):
        global loaded_games
        row = loaded_games // 3

        next_set_of_games = recommended_games.iloc[loaded_games:loaded_games + 12]

        col = 0
        for _, game in next_set_of_games.iterrows():
            game_name = game['name']
            game_score = game['Recommendation Score']
            header_image_url = df[df['name'] == game['name']]['header_image'].values[0] if 'header_image' in df.columns else None
            game_website = df[df['name'] == game_name]['website'].values[0] if 'website' in df.columns else None

            game_image = get_image(header_image_url)
            if game_image:
                game_frame = ctk.CTkFrame(scrollable_frame, fg_color="#2a2d2e", corner_radius=0)
                game_frame.grid(row=row, column=col % 3, padx=5, pady=5)
                col += 1

                image_label = tk.Label(game_frame, image=game_image, bg="#2a2d2e")
                image_label.image = game_image
                image_label.pack()

                if game_website:
                    image_label.bind("<Button-1>", lambda e, url=game_website: webbrowser.open_new(url))

                name_label = tk.Label(
                    game_frame, 
                    text=game_name, 
                    font=("Arial", 12), 
                    highlightthickness=0, 
                    wraplength=240,
                    justify="left",
                    bg="#2a2d2e",
                    fg="white"
                )

                name_label.place(relx=1, rely=0.85, anchor="se")

                score_label = tk.Label(
                    game_frame, 
                    text=f"Recommendation Score: {game_score}%",
                    font=("Arial", 10), 
                    highlightthickness=0, 
                    bg="#2a2d2e",
                    fg="white"
                )
                score_label.place(relx=1, rely=1, anchor="se") 

            if col % 3 == 0:
                row += 1

        loaded_games += len(next_set_of_games)

        button.grid(row=row + 1, column=0, columnspan=3, pady=(15, 15))

    load_more_games(see_more_button)

    see_more_button.grid(row=(loaded_games // 3), column=0, columnspan=3, pady=(15, 15))

### Recommendation Score Calculations ####################################################

def recommender(age, budget, languages, play_history, model_svd, game_encoder, n=10):
    # Get a list of games the user has played from the play history
    played_games = [game for game, _ in play_history] 
    # Filter the games based on the criteria provided
    filtered_games = df[df['required_age'] <= age]
    filtered_games = filtered_games[filtered_games['price'] <= budget]
    filtered_games = filtered_games[filtered_games['supported_languages'].apply(lambda x: any(lang in x for lang in languages))]  # Language filter

    content_based_recommendations = calculate_similarity_scores(filtered_games, play_history)

    if 'Recommendation Score' not in content_based_recommendations.columns:
        content_based_recommendations['Recommendation Score'] = 0

    # Fetch top games from the trained model (collaborative filtering)
    model_based_recommendations = calculate_content_based_score(play_history, model_svd, recommendations_df, n)

    model_boosted_recs = pd.DataFrame({'name': model_based_recommendations})
    model_boosted_recs['Boost'] = np.linspace(10, 1, len(model_boosted_recs))
    model_boosted_recs['Boost'] = model_boosted_recs['Boost'] / 100

    content_based_recommendations['Content Contribution'] = 0.0

    # Merge model-based recommendations with content-based recommendations
    content_based_recommendations = content_based_recommendations.merge(model_boosted_recs, on='name', how='left')
    content_based_recommendations['Boost'] = content_based_recommendations['Boost'].fillna(0)
    content_based_recommendations['Boost'] = content_based_recommendations['Boost'] * 2

    # Set Content Contribution to the actual boost percentage for games that are boosted
    content_based_recommendations['Content Contribution'] = (content_based_recommendations['Recommendation Score'] * (1 + content_based_recommendations['Boost']) - content_based_recommendations['Recommendation Score']).round(0)

    # Apply the boost to the final recommendation score
    content_based_recommendations['Recommendation Score'] = (content_based_recommendations['Recommendation Score'] * (1 + content_based_recommendations['Boost'])).round(0)

    content_based_recommendations = content_based_recommendations[~content_based_recommendations['name'].isin(played_games)]

    final_recommendations = content_based_recommendations.groupby('name', as_index=False).agg({
        'Recommendation Score': 'sum',
        'Content Contribution': 'sum',
        'Similarity Contribution': 'sum',
        'Popularity Contribution': 'sum',
    }).sort_values(by='Recommendation Score', ascending=False)

    top_10_games = final_recommendations.head(10)
    for idx, game in top_10_games.iterrows():
        print(f"Game: {game['name']}")
        print(f" - Recommendation Score: {game['Recommendation Score']}")
        print(f" - Collaborative Contribution: {game['Content Contribution']:.2f}%")
        print(f" - Similarity Contribution: {game['Similarity Contribution']:.2f}%")
        print(f" - Popularity Contribution: {game['Popularity Contribution']:.2f}%")
        print("-" * 40)

    return final_recommendations


def calculate_similarity_scores(filtered_games, play_history):
    # Extract textual features using TF-IDF for 'tags', 'categories', 'genres', and 'name'
    tfidf = TfidfVectorizer(stop_words=None, max_features=500)  # Limiting to top 500 features

    # Give priority to 'tags', then 'categories', 'genres', and also include 'name'
    filtered_games['combined_text'] = (
        filtered_games['tags'].fillna('') * 3 + " " +
        filtered_games['categories'].fillna('') * 2 + " " +
        filtered_games['genres'].fillna('') + " "
    )

    # Apply TF-IDF on the combined text data
    text_features = tfidf.fit_transform(filtered_games['combined_text'])  # Handle missing values with empty string

    # Calculate similarity scores (cosine similarity)
    similarity_matrix = cosine_similarity(text_features)

    def ngrams(string, n):
        string = string.lower()
        return set([string[i:i+n] for i in range(len(string)-n+1)])

    def get_title_similarity(game_name1, game_name2, n=3):
        ngrams1 = ngrams(game_name1, n)
        ngrams2 = ngrams(game_name2, n)
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0

    # Apply similarity score calculation to each game
    filtered_games = filtered_games.reset_index(drop=True)

    total_hours = sum([int(hours) for _, hours in play_history])

    def get_similarity_to_played_games(game_idx):
        similarity_scores = []
        for played_game_name, hours in play_history:
            if played_game_name in filtered_games['name'].values:
                played_game_indices = filtered_games.index[filtered_games['name'] == played_game_name].tolist()

                # Only calculate if the game is in the filtered dataset
                if played_game_indices:
                    played_game_index = played_game_indices[0]
                    if game_idx < similarity_matrix.shape[0] and played_game_index < similarity_matrix.shape[0]:
                        # Combine cosine similarity and title similarity for better scoring
                        cosine_sim = similarity_matrix[game_idx, played_game_index]
                        title_sim = get_title_similarity(
                            filtered_games.loc[game_idx, 'name'],
                            filtered_games.loc[played_game_index, 'name']
                        )
                        combined_score = (0.6 * cosine_sim) + (0.4 * title_sim)
                        hours_weight = int(hours) / total_hours
                        weighted_score = combined_score * hours_weight

                        similarity_scores.append(weighted_score)

        return round(np.mean(similarity_scores) * 100) if similarity_scores else 0

    # Apply similarity score calculation to each game
    filtered_games['Similarity Score'] = [
        get_similarity_to_played_games(idx) for idx in range(len(filtered_games))
    ]

    # Boost popularity based on 'peak_ccu' using log1p for a better spread
    max_ccu = filtered_games['peak_ccu'].max()

    def calculate_popularity_boost(peak_ccu):
        return np.log1p(peak_ccu) / np.log1p(max_ccu) if max_ccu else 0

    filtered_games['Popularity Boost'] = filtered_games['peak_ccu'].apply(calculate_popularity_boost)

    # ombine the similarity score and popularity boost
    similarity_weight = 0.85
    popularity_weight = 0.15
    filtered_games['Final Score'] = (
        similarity_weight * filtered_games['Similarity Score'] +
        popularity_weight * filtered_games['Popularity Boost'] * 100
    )

    filtered_games['Recommendation Score'] = filtered_games['Final Score'].round(0)

    # Sort games by the final score in descending order
    top_10_games = filtered_games.sort_values(by='Final Score', ascending=False).head(10)

    filtered_games['Similarity Contribution'] = 0.0
    filtered_games['Popularity Contribution'] = 0.0

    for idx, game in filtered_games.iterrows():
        # Calculate contribution percentages
        similarity_contribution_percentage = (similarity_weight * game['Similarity Score']) / game['Final Score'] * 100
        popularity_contribution_percentage = (popularity_weight * game['Popularity Boost'] * 100) / game['Final Score'] * 100

        similarity_contribution_value = (similarity_contribution_percentage / 100) * game['Final Score']
        popularity_contribution_value = (popularity_contribution_percentage / 100) * game['Final Score']

        filtered_games.at[idx, 'Similarity Contribution'] = similarity_contribution_value
        filtered_games.at[idx, 'Popularity Contribution'] = popularity_contribution_value

    for idx, game in top_10_games.iterrows():
        similarity_contribution_value = filtered_games.at[idx, 'Similarity Contribution']
        popularity_contribution_value = filtered_games.at[idx, 'Popularity Contribution']

    return filtered_games[['name', 'Recommendation Score', 'Similarity Contribution', 'Popularity Contribution']]


def calculate_content_based_score(play_history, model_svd, recommendations_df, n=5):
    input_game_ids = []
    for game, _ in play_history:
        # Find the app_id for the given game
        app_id_row = recommendations_df[recommendations_df['name'] == game]
        if not app_id_row.empty:
            input_game_id = app_id_row['app_id'].values[0]
            input_game_ids.append(input_game_id)
        else:
            print(f"Game '{game}' not found in dataset.")

    # Find users who have played the input games
    users_who_played_input_games = recommendations_df[recommendations_df['app_id'].isin(input_game_ids)]['user_id'].unique()

    # Find other games these users have played (excluding the input games)
    relevant_games = recommendations_df[(recommendations_df['user_id'].isin(users_who_played_input_games)) & (~recommendations_df['app_id'].isin(input_game_ids))]

    games_to_predict = relevant_games['app_id'].unique()

    #Predict ratings for the relevant games using the SVD model
    game_predictions = []
    for game_id in games_to_predict:
        prediction = model_svd.predict(0, game_id)  # Use user_id=0 since we're focusing on item recommendations
        game_predictions.append((game_id, prediction.est))

    # Sort the predicted ratings in descending order and select the top `n`
    top_n_recommendations = sorted(game_predictions, key=lambda x: x[1], reverse=True)[:n]
    print(f"Top {n} recommendations (app_id): {top_n_recommendations}")

    top_n_game_ids = [game_id for game_id, _ in top_n_recommendations]
    top_n_games = recommendations_df[recommendations_df['app_id'].isin(top_n_game_ids)]['name'].unique()

    print(f"Top recommended games: {top_n_games}")

    return top_n_games

# Start with the age screen
age_screen()

app.mainloop()