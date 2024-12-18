# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

def load_data():
    """Load and combine data from all CSV files in the data/raw directory"""
    data_path = Path("data/raw")
    dfs = []
    
    # Add debug information
    st.sidebar.subheader("Data Loading Status")
    
    for csv_file in data_path.glob("arm_angles_*.csv"):
        year = csv_file.stem.split('_')[-1]  # Extract year from filename
        try:
            df = pd.read_csv(csv_file)
            n_pitchers = df['pitcher'].nunique()
            st.sidebar.write(f"✓ {csv_file.name}: {len(df)} rows, {n_pitchers} pitchers")
            dfs.append(df)
        except Exception as e:
            st.sidebar.error(f"Error loading {csv_file.name}: {str(e)}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else None

def main():
    st.title("Pitcher Arm Angle Analysis")
    
    # Load data from repository
    with st.spinner("Loading data..."):
        df = load_data()
        
    if df is not None:
        # Data validation checks
        st.subheader("Data Validation")
        
        # Check for required columns
        required_columns = ['pitcher', 'pitcher_name', 'year', 'pitch_hand', 
                          'n_pitches', 'ball_angle', 'relative_release_ball_x', 
                          'release_ball_z']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return
        
        # Display basic data statistics
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
        
        # Show data distribution by year
        yearly_counts = df.groupby('year').agg({
            'pitcher': 'nunique',
            'n_pitches': 'sum'
        }).reset_index()
        
        yearly_counts.columns = ['Year', 'Number of Pitchers', 'Total Pitches']
        st.subheader("Data Distribution by Year")
        st.dataframe(yearly_counts)
        
        # Sample of raw data
        st.subheader("Sample of Raw Data")
        st.dataframe(df.head())
        
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
        
        # Check for outliers or unusual values
        st.subheader("Data Range Check")
        numeric_cols = ['ball_angle', 'relative_release_ball_x', 'release_ball_z', 'n_pitches']
        stats_df = df[numeric_cols].describe()
        st.dataframe(stats_df)
        
        # Allow downloading the combined dataset
        st.download_button(
            label="Download combined data as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='combined_arm_angles.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
