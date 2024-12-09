import pandas as pd
import codecs

# load data
with codecs.open('tbc.csv', 'r', errors='replace') as file:
    df = pd.read_csv(file, index_col=False)

df_clean = df.copy()
df_clean['peak_ccu'] = df_clean['peak_ccu'].astype(str)

# clean the 'peak ccu' column
df_clean['peak_ccu'] = df_clean['peak_ccu'].str.replace(r'[,]', '', regex=True).str.extract(r'(\d+)').astype(float)

# drop any rows with missing 'peak ccu' values
df_clean = df_clean.dropna(subset=['peak_ccu'])

df_clean = df_clean.sort_values(by='peak_ccu', ascending=False)

# select the top 15% of the games with the highest peak ccu
top_15_percent = 10000
df_top_15_percent = df_clean.head(top_15_percent)

columns_to_keep = [
    'name', 'required_age', 'price', 
    'about_the_game', 'reviews', 
    'header_image', 'website', 'windows', 
    'mac', 'linux', 'recommendations', 'supported_languages',  
    'developers', 'publishers', 'categories', 'genres', 'positive', 
    'negative', 'estimated_owners', 'average_playtime_forever', 
    'peak_ccu', 'tags', 'pct_pos_total', 'num_reviews_total'
]

df_top_15_percent = df_top_15_percent[columns_to_keep]

df_top_15_percent.to_csv('cleaned_games.csv', index=False)

print("Shape of the cleaned and filtered dataset:", df_top_15_percent.shape)