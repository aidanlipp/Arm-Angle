import pandas as pd
import requests
import time
import json
from datetime import datetime

def fetch_pitcher_stats(player_id, year):
    """
    Fetch pitching stats from Baseball Savant's statcast search
    """
    url = f"https://baseballsavant.mlb.com/savant-player/{player_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Add delay to respect rate limits
        time.sleep(1)
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch data for player {player_id}: Status code {response.status_code}")
            return None

        # Try to find the stats section in the response
        if 'stats_sortable' in response.text:
            # Extract the JSON data from the page
            start_marker = "var stats_sortable = "
            end_marker = ";</script>"
            start_idx = response.text.find(start_marker) + len(start_marker)
            end_idx = response.text.find(end_marker, start_idx)
            
            if start_idx > -1 and end_idx > -1:
                stats_json = response.text[start_idx:end_idx]
                stats_data = json.loads(stats_json)
                
                # Find the relevant year's data
                year_data = next((season for season in stats_data if str(season.get('year')) == str(year)), None)
                
                if year_data:
                    return {
                        'player_id': player_id,
                        'year': year,
                        'games': year_data.get('g', 0),
                        'games_started': year_data.get('gs', 0)
                    }
        
        return None
        
    except Exception as e:
        print(f"Error fetching stats for player {player_id}: {str(e)}")
        return None

def process_and_classify_data(df, min_games=10):
    """
    Process the arm angles data and add role classification
    """
    results = []
    total_pitchers = len(df['pitcher'].unique())
    
    # Create progress bar
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for idx, (player_id, player_data) in enumerate(df.groupby('pitcher')):
        # Update progress
        progress = (idx + 1) / total_pitchers
        progress_bar.progress(progress)
        progress_text.text(f"Processing pitcher {idx + 1} of {total_pitchers}")
        
        # Get player name from the data
        player_name = player_data['pitcher_name'].iloc[0]
        
        # Fetch stats for each year
        for year in player_data['year'].unique():
            stats = fetch_pitcher_stats(player_id, year)
            
            if stats and stats['games'] >= min_games:
                # Calculate starter percentage
                starter_pct = stats['games_started'] / stats['games']
                role = 'Starter' if starter_pct >= 0.6 else 'Reliever'
                
                # Add to results
                results.append({
                    'pitcher': player_id,
                    'pitcher_name': player_name,
                    'year': year,
                    'games': stats['games'],
                    'games_started': stats['games_started'],
                    'starter_pct': starter_pct,
                    'role': role
                })
    
    # Clear progress indicators
    progress_bar.empty()
    progress_text.empty()
    
    # Convert results to DataFrame
    roles_df = pd.DataFrame(results)
    
    # Merge with original data
    return df.merge(roles_df[['pitcher', 'year', 'role', 'games', 'games_started']], 
                   on=['pitcher', 'year'], 
                   how='left')

def analyze_distributions(df):
    """
    Create visualizations for arm angle distributions by role and handedness
    """
    # Create combined category
    df['pitcher_type'] = df['pitch_hand'] + ' ' + df['role']
    
    # Create distribution plot
    fig = px.histogram(
        df,
        x="ball_angle",
        color="pitcher_type",
        title="Distribution of Arm Angles by Pitcher Type",
        labels={"ball_angle": "Arm Angle", "count": "Number of Pitchers"},
        barmode="overlay",
        opacity=0.7,
        category_orders={
            "pitcher_type": ["R Starter", "R Reliever", "L Starter", "L Reliever"]
        }
    )
    
    return fig

def main():
    """
    Main function to run the analysis
    """
    df = load_data()  # Your existing load_data function
    
    if df is not None:
        st.title("Pitcher Arm Angle Analysis")
        
        # Process and classify data
        with st.spinner("Fetching and classifying pitcher roles..."):
            classified_df = process_and_classify_data(df)
        
        # Create and show distributions
        fig = analyze_distributions(classified_df)
        st.plotly_chart(fig)
        
        # Show summary statistics
        st.subheader("Summary Statistics by Pitcher Type")
        stats = classified_df.groupby(['pitch_hand', 'role']).agg({
            'ball_angle': ['count', 'mean', 'std'],
            'pitcher': 'nunique'
        }).round(2)
        st.dataframe(stats)
        
        # Option to download processed data
        st.download_button(
            label="Download processed data",
            data=classified_df.to_csv(index=False).encode('utf-8'),
            file_name="pitcher_classifications.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
