import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from utils.data_processing import process_arm_angles, fetch_player_stats

def load_data():
    """Load and combine data from all CSV files in the data/raw directory"""
    data_path = Path("data/raw")
    dfs = []
    
    for csv_file in data_path.glob("arm_angles_*.csv"):
        year = csv_file.stem.split('_')[-1]  # Extract year from filename
        df = pd.read_csv(csv_file)
        if 'year' not in df.columns:
            df['year'] = year
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else None

def main():
    st.title("Pitcher Arm Angle Analysis")
    
    # Load data from repository
    with st.spinner("Loading data..."):
        df = load_data()
        
    if df is not None:
        # Convert year to string for better display
        df['year'] = df['year'].astype(str)
        
        # Basic data overview
        st.subheader("Data Overview")
        st.write(f"Number of unique pitchers: {df['pitcher'].nunique()}")
        st.write(f"Years of data: {df['year'].min()} - {df['year'].max()}")
        
        # Add filters
        st.sidebar.header("Filters")
        min_pitches = st.sidebar.slider(
            "Minimum Pitches",
            min_value=0,
            max_value=int(df['n_pitches'].max()),
            value=1000
        )
        
        selected_years = st.sidebar.multiselect(
            "Select Years",
            options=sorted(df['year'].unique()),
            default=sorted(df['year'].unique())
        )
        
        pitch_hand = st.sidebar.multiselect(
            "Pitcher Handedness",
            options=sorted(df['pitch_hand'].unique()),
            default=sorted(df['pitch_hand'].unique())
        )
        
        # Apply filters
        filtered_df = df[
            (df['n_pitches'] >= min_pitches) &
            (df['year'].isin(selected_years)) &
            (df['pitch_hand'].isin(pitch_hand))
        ]
        
        # Create visualization of arm angle distribution
        fig_dist = px.histogram(
            filtered_df,
            x="ball_angle",
            color="year",
            title="Distribution of Pitcher Arm Angles by Year",
            labels={"ball_angle": "Arm Angle", "count": "Number of Pitchers"},
            barmode="overlay",
            opacity=0.7
        )
        st.plotly_chart(fig_dist)
        
        # Show average arm angle trends over time
        yearly_avg = filtered_df.groupby(['year', 'pitch_hand'])['ball_angle'].mean().reset_index()
        fig_trend = px.line(
            yearly_avg,
            x="year",
            y="ball_angle",
            color="pitch_hand",
            title="Average Arm Angle Over Time by Handedness",
            labels={"ball_angle": "Average Arm Angle", "year": "Year"},
            markers=True
        )
        st.plotly_chart(fig_trend)
        
        # Allow user to select specific pitchers to analyze
        selected_pitchers = st.multiselect(
            "Select pitchers to analyze",
            options=sorted(filtered_df['pitcher_name'].unique())
        )
        
        if selected_pitchers:
            pitcher_data = filtered_df[filtered_df['pitcher_name'].isin(selected_pitchers)]
            
            # Create line plot for selected pitchers
            fig_pitchers = px.line(
                pitcher_data,
                x="year",
                y="ball_angle",
                color="pitcher_name",
                title="Arm Angle Trends for Selected Pitchers",
                labels={"ball_angle": "Arm Angle", "year": "Year"},
                markers=True
            )
            st.plotly_chart(fig_pitchers)
            
            # Show detailed stats for selected pitchers
            st.subheader("Detailed Statistics")
            stats_df = pitcher_data.groupby(['pitcher_name', 'year', 'pitch_hand']).agg({
                'ball_angle': 'mean',
                'n_pitches': 'sum',
                'relative_release_ball_x': 'mean',
                'release_ball_z': 'mean'
            }).round(2)
            st.dataframe(stats_df)
    
    else:
        st.error("No data files found in data/raw directory. Please ensure CSV files are present.")

if __name__ == "__main__":
    main()
