import pandas as pd

# load csv and use chunking
file_path = 'filtered_recommendations.csv'
chunk_size = 100000  # Adjust based on your memory capacity
output_file_path = 'filtered_recommendations_final.csv'

# create an empty DataFrame
df_filtered_combined = pd.DataFrame()

# Process the CSV in chunks
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    chunk_filtered = chunk[(chunk['hours'] > 250) & (chunk['is_recommended'] == True)]

    # remove the unwanted columns
    columns_to_drop = ['helpful', 'funny', 'date', 'review_id']
    chunk_filtered = chunk_filtered.drop(columns=columns_to_drop)

    df_filtered_combined = pd.concat([df_filtered_combined, chunk_filtered])

user_counts = df_filtered_combined['user_id'].value_counts()
df_filtered_final = df_filtered_combined[df_filtered_combined['user_id'].isin(user_counts[user_counts > 1].index)]

df_filtered_final.to_csv(output_file_path, index=False)

print(f"Filtered and cleaned data saved to {output_file_path}")
