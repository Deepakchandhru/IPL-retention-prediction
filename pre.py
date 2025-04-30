import pandas as pd

# Load the dataset
df = pd.read_csv('all_matches.csv')  # Replace with your actual dataset path

# Clean column names
df.columns = df.columns.str.strip()

# Combine venue columns into a single column if there are duplicat

# Convert necessary columns to proper datatypes
# Extract last two characters of the Season and convert to int
df['Season'] = df['Season'].astype(str).str[-2:].astype(int)
df['Runs Off Bat'] = df['Runs Off Bat'].astype(int)
df['Runs Conceed by Bowler'] = df['Runs Conceed by Bowler'].astype(int)
df['Is Wicket By Bowler'] = df['Is Wicket By Bowler'].astype(int)

# Define home ground performance
home_ground_stats = df.groupby(['Striker', 'Season', 'Venue']).agg(
    home_runs=('Runs Off Bat', 'sum')
).reset_index()

home_bowling_stats = df.groupby(['Bowler', 'Season', 'Venue']).agg(
    home_wickets=('Is Wicket By Bowler', 'sum')
).reset_index()

# Aggregate batting stats
batting_stats = df.groupby(['Striker', 'Season','Venue','Batting Team']).agg(
    total_runs=('Runs Off Bat', 'sum'),
    balls_faced=('Ball', 'count'),
    strike_rate=('Runs Off Bat', lambda x: (x.sum() / x.count()) * 100)
).reset_index()

# Aggregate bowling stats
bowling_stats = df.groupby(['Bowler', 'Season','Venue','Bowling Team']).agg(
    total_wickets=('Is Wicket By Bowler', 'sum'),
    total_runs_conceded=('Runs Conceed by Bowler', 'sum'),
    balls_bowled=('Ball', 'count'),
    economy_rate=('Runs Conceed by Bowler', lambda x: x.sum() * 6 / x.count())
).reset_index()

# Merge batting and bowling stats
player_stats = pd.merge(batting_stats, bowling_stats, left_on=['Striker', 'Season','Venue'], right_on=['Bowler', 'Season','Venue'], how='outer')
player_stats.fillna(0, inplace=True) 

# Save processed data
player_stats.to_csv('processed_ipl.csv', index=False)

print("Data processing complete! Saved as 'processed_ipl_data.csv'")
