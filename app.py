# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import requests
import time
import json
from bs4 import BeautifulSoup

def load_arm_angle_data():
    """Load and combine data from all CSV files in the data/raw directory"""
    data_path = Path("data/raw")
    dfs = []
    
    # List all files in directory
    st.sidebar.write("Loading files from:", data_path)
    
    for csv_file in data_path.glob("ArmAngles*.csv"):
        year = csv_file.stem[-2:]  # Extract year (20, 21, etc.)
        full_year = f"20{year}"    # Convert to full year (2020, 2021, etc.)
        
        try:
            df = pd.read_csv(csv_file)
            n_pitchers = df['pitcher'].nunique()
            st.sidebar.success(f"✓ Loaded {csv_file.name}: {len(df)} rows, {n_pitchers} pitchers")
            
            # Make sure we have a year column
            if 'year' not in df.columns:
                df['year'] = full_year
                
            dfs.append(df)
        except Exception as e:
            st.sidebar.error(f"Error loading {csv_file.name}: {str(e)}")
    
    if not dfs:
        st.sidebar.error("No data files were loaded!")
        return None
        
    return pd.concat(dfs, ignore_index=True)

def fetch_pitcher_stats(player_id, year):
    """Fetch pitching stats from Baseball Savant"""
    # Use statcast search endpoint instead of player page
    url = f"https://baseballsavant.mlb.com/statcast_search/csv?all=true&player_id={player_id}&year={year}&game_type=R"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        time.sleep(1)  # Rate limiting
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            # Parse CSV data
            df = pd.read_csv(pd.StringIO(response.text))
            
            # Get games and games started
            games = df['game_pk'].nunique()
            games_started = df[df['inning'] == 1].groupby('game_pk')['game_pk'].first().count()
            
            return {
                'player_id': player_id,
                'year': year,
                'games': games,
                'games_started': games_started,
                'role': 'Starter' if (games_started / games >= 0.6) else 'Reliever' if games > 0 else 'Unknown'
            }
            
        return None
    except Exception as e:
        st.warning(f"Error fetching stats for player {player_id}: {str(e)}")
        return None

def main():
    st.title("Pitcher Arm Angle Analysis")
    
    # Load the arm angle data
    df = load_arm_angle_data()
    
    if df is not None:
        # Convert year to string for better display
        df['year'] = df['year'].astype(str)
        
        # Basic data overview
        st.header("Data Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
            st.metric("Unique Pitchers", df['pitcher'].nunique())
        
        with col2:
            st.metric("Year Range", f"{df['year'].min()} - {df['year'].max()}")
            st.metric("Left-handed Pitchers", len(df[df['pitch_hand'] == 'L']['pitcher'].unique()))
        
        with col3:
            st.metric("Average Pitches", int(df['n_pitches'].mean()))
            st.metric("Right-handed Pitchers", len(df[df['pitch_hand'] == 'R']['pitcher'].unique()))
        
        # Add a button to start scraping
        if st.button("Fetch Pitcher Role Data"):
            roles_data = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get unique pitcher-year combinations
            pitcher_years = df[['pitcher', 'year', 'pitcher_name']].drop_duplicates()
            total_pitchers = len(pitcher_years)
            
            for idx, row in pitcher_years.iterrows():
                progress = (idx + 1) / total_pitchers
                progress_bar.progress(progress)
                status_text.text(f"Processing pitcher {idx + 1} of {total_pitchers}: {row['pitcher_name']}")
                
                stats = fetch_pitcher_stats(row['pitcher'], row['year'])
                if stats:
                    roles_data.append(stats)
                    st.write(f"Retrieved stats for {row['pitcher_name']}: {stats['games']} games, {stats['games_started']} starts - {stats['role']}")
            
            # Create roles DataFrame
            roles_df = pd.DataFrame(roles_data)
            
            # Merge with original data
            df = df.merge(roles_df[['player_id', 'year', 'role']], 
                         left_on=['pitcher', 'year'],
                         right_on=['player_id', 'year'],
                         how='left')
            
            # Update visualization with roles
            fig = px.histogram(
                df,
                x="ball_angle",
                color="role",
                facet_col="pitch_hand",
                title="Distribution of Arm Angles by Role and Handedness",
                labels={"ball_angle": "Arm Angle", "count": "Number of Pitchers"},
                barmode="overlay",
                opacity=0.7
            )
            st.plotly_chart(fig)
            
            # Show role statistics
            st.header("Role Statistics")
            role_stats = df.groupby(['pitch_hand', 'role'])['ball_angle'].agg(['count', 'mean', 'std']).round(2)
            st.dataframe(role_stats)
            
        else:
            # Show basic distribution without roles
            fig = px.histogram(
                df,
                x="ball_angle",
                color="pitch_hand",
                title="Distribution of Arm Angles by Handedness",
                labels={"ball_angle": "Arm Angle", "count": "Number of Pitchers"},
                barmode="overlay",
                opacity=0.7
            )
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
