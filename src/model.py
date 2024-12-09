import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from sklearn.preprocessing import LabelEncoder
import pickle

# function to see progress
class SVDWithProgress(SVD):
    def fit(self, trainset):
        print(f"Training SVD with {self.n_epochs} epochs...")
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}...")
            super().fit(trainset)
        return self

# load game dataset
df = pd.read_csv("../data/recommendations.csv")

# encode user_id and game_id for surprise library
user_encoder = LabelEncoder()
game_encoder = LabelEncoder()

df['user_id'] = user_encoder.fit_transform(df['user_id'])
df['game_id'] = game_encoder.fit_transform(df['name'])

# use the 'hours' column as the target value
df['rating'] = df['hours'] 
reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
data = Dataset.load_from_df(df[['user_id', 'game_id', 'rating']], reader)

# train/test split
trainset = data.build_full_trainset()

model_svd = SVDWithProgress(n_epochs=10)
model_svd.fit(trainset)

# save the model and encoders
with open('svd_model.pkl', 'wb') as model_file:
    pickle.dump(model_svd, model_file)
with open('game_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(game_encoder, encoder_file)

# function to recommend games based on games you've played
def recommend_based_on_games(input_games, model_svd, df, game_encoder, n=5):
    # encode the input game names to match game IDs in the dataset
    input_game_ids = game_encoder.transform(input_games)
    
    # find users who have played the input games
    users_who_played_input_games = df[df['game_id'].isin(input_game_ids)]['user_id'].unique()
    
    # find games these users have played, excluding the input games
    relevant_games = df[(df['user_id'].isin(users_who_played_input_games)) & (~df['game_id'].isin(input_game_ids))]

    games_to_predict = relevant_games['game_id'].unique()

    # predict ratings for these games based on collaborative filtering (SVD)
    game_predictions = []
    for game_id in games_to_predict:
        prediction = model_svd.predict(0, game_id)
        game_predictions.append((game_id, prediction.est))

    # sort the predicted hours in descending order
    top_n_recommendations = sorted(game_predictions, key=lambda x: x[1], reverse=True)[:n]

    top_n_game_ids = [game_id for game_id, _ in top_n_recommendations]
    top_n_games = game_encoder.inverse_transform(top_n_game_ids)

    return top_n_games

def load_model_and_encoders():
    with open('svd_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open('game_encoder.pkl', 'rb') as encoder_file:
        loaded_game_encoder = pickle.load(encoder_file)
    return loaded_model, loaded_game_encoder

# load the model and encoders
loaded_model_svd, loaded_game_encoder = load_model_and_encoders()

input_games = ['Dota 2', 'Europa Universalis IV']

recommended_games = recommend_based_on_games(input_games, loaded_model_svd, df, loaded_game_encoder)
print(f"Recommended games based on {input_games}:")
for i, game in enumerate(recommended_games, 1):
    print(f"{i}. {game}")
