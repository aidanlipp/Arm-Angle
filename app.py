# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import requests
import time
import json

def load_data():
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
    url = f"https://baseballsavant.mlb.com/savant-player/{player_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        time.sleep(1)  # Rate limiting
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            # Process the response here
            return {
                'player_id': player_id,
                'year': year,
                'games': 0,  # Placeholder
                'games_started': 0  # Placeholder
            }
        return None
    except Exception as e:
        st.error(f"Error fetching stats for player {player_id}: {str(e)}")
        return None

def main():
    st.title("Pitcher Arm Angle Analysis")
    
    # Load the arm angle data
    df = load_data()
    
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
            with st.spinner("Fetching pitcher statistics..."):
                # Get a sample of pitchers to test
                sample_pitchers = df[['pitcher', 'year']].drop_duplicates().head(5)
                
                for _, row in sample_pitchers.iterrows():
                    stats = fetch_pitcher_stats(row['pitcher'], row['year'])
                    if stats:
                        st.write(f"Retrieved stats for pitcher {row['pitcher']}")
        
        # Basic distribution visualization
        st.header("Arm Angle Distribution")
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
        
        # Yearly statistics
        st.header("Yearly Statistics")
        yearly_stats = df.groupby('year').agg({
            'pitcher': 'nunique',
            'n_pitches': 'sum',
            'ball_angle': ['mean', 'std']
        }).round(2)
        yearly_stats.columns = ['Number of Pitchers', 'Total Pitches', 'Avg Arm Angle', 'Std Arm Angle']
        st.dataframe(yearly_stats)

if __name__ == "__main__":
    main()
