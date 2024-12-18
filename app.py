# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

def load_data():
    """Load and combine data from all CSV files in the data/raw directory"""
    # Print current working directory
    st.sidebar.write("Current working directory:", os.getcwd())
    
    # Construct data path
    data_path = Path("data/raw")
    st.sidebar.write("Looking for data in:", data_path.absolute())
    
    # Check if directory exists
    if not data_path.exists():
        st.sidebar.error(f"Directory not found: {data_path.absolute()}")
        return None
    
    # List all files in directory
    st.sidebar.write("Files found in directory:")
    for file in data_path.glob("*"):
        st.sidebar.write(f"- {file.name}")
    
    dfs = []
    # Updated file pattern to match your naming convention
    for csv_file in data_path.glob("ArmAngles*.csv"):
        # Extract year from filename (20, 21, etc.)
        year = csv_file.stem[-2:]  
        full_year = f"20{year}"  # Convert to full year (2020, 2021, etc.)
        
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
        
    combined_df = pd.concat(dfs, ignore_index=True)
    st.sidebar.success(f"Successfully loaded {len(dfs)} files with {len(combined_df)} total rows")
    return combined_df

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
        
        # Distribution of arm angles
        st.subheader("Arm Angle Distribution")
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
        
        # Show data statistics
        st.subheader("Yearly Statistics")
        yearly_stats = df.groupby('year').agg({
            'pitcher': 'nunique',
            'n_pitches': 'sum',
            'ball_angle': ['mean', 'std']
        }).round(2)
        yearly_stats.columns = ['Number of Pitchers', 'Total Pitches', 'Avg Arm Angle', 'Std Arm Angle']
        st.dataframe(yearly_stats)

if __name__ == "__main__":
    main()
