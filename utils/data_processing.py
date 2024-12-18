# utils/data_processing.py
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

def process_arm_angles(df):
    """
    Process the arm angle data and calculate additional metrics
    """
    # Add any additional processing needed for arm angles
    df['arm_angle_category'] = pd.cut(
        df['ball_angle'],
        bins=[-90, 0, 30, 60, 90],
        labels=['Submarine', 'Sidearm', 'Three-quarter', 'Over-the-top']
    )
    
    return df

def fetch_player_stats(player_id, year):
    """
    Fetch additional statistics from Baseball Savant for a given player
    """
    base_url = f"https://baseballsavant.mlb.com/savant-player/{player_id}?stats={year}"
    
    try:
        # Add delay to respect rate limits
        time.sleep(1)
        
        response = requests.get(base_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract relevant statistics
        # This is a placeholder - we'll need to adjust based on the actual HTML structure
        stats = {
            'player_id': player_id,
            'year': year,
            # Add more stats as needed
        }
        
        return stats
    
    except Exception as e:
        print(f"Error fetching stats for player {player_id}: {str(e)}")
        return None

def enrich_data_with_stats(df):
    """
    Enrich the arm angle data with additional statistics
    """
    enriched_data = []
    
    for _, row in df.iterrows():
        stats = fetch_player_stats(row['pitcher'], row['year'])
        if stats:
            enriched_row = {**row, **stats}
            enriched_data.append(enriched_row)
    
    return pd.DataFrame(enriched_data)
