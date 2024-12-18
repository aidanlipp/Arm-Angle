# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import process_arm_angles, fetch_player_stats

def main():
    st.title("Pitcher Arm Angle Analysis")
    
    # File uploaders for each year
    st.sidebar.header("Upload Data Files")
    uploaded_files = {}
    years = range(2020, 2025)
    
    for year in years:
        uploaded_file = st.sidebar.file_uploader(f"Upload {year} Data", type=['csv'], key=f"file_{year}")
        if uploaded_file is not None:
            uploaded_files[year] = uploaded_file
    
    if uploaded_files:
        # Process the data
        with st.spinner("Processing arm angle data..."):
            # Combine all years of data
            dfs = []
            for year, file in uploaded_files.items():
                df = pd.read_csv(file)
                df['year'] = year
                dfs.append(df)
            
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Basic data overview
            st.subheader("Data Overview")
            st.write(f"Number of unique pitchers: {combined_df['pitcher'].nunique()}")
            st.write(f"Years of data: {combined_df['year'].min()} - {combined_df['year'].max()}")
            
            # Create visualization of arm angle distribution
            fig_dist = px.histogram(
                combined_df,
                x="ball_angle",
                color="year",
                title="Distribution of Pitcher Arm Angles by Year",
                labels={"ball_angle": "Arm Angle", "count": "Number of Pitchers"},
                barmode="overlay",
                opacity=0.7
            )
            st.plotly_chart(fig_dist)
            
            # Show average arm angle trends over time
            yearly_avg = combined_df.groupby('year')['ball_angle'].mean().reset_index()
            fig_trend = px.line(
                yearly_avg,
                x="year",
                y="ball_angle",
                title="Average Arm Angle Over Time",
                labels={"ball_angle": "Average Arm Angle", "year": "Year"}
            )
            st.plotly_chart(fig_trend)
            
            # Allow user to select specific pitchers to analyze
            selected_pitchers = st.multiselect(
                "Select pitchers to analyze",
                options=combined_df['pitcher_name'].unique()
            )
            
            if selected_pitchers:
                pitcher_data = combined_df[combined_df['pitcher_name'].isin(selected_pitchers)]
                
                # Create line plot for selected pitchers
                fig_pitchers = px.line(
                    pitcher_data,
                    x="year",
                    y="ball_angle",
                    color="pitcher_name",
                    title="Arm Angle Trends for Selected Pitchers",
                    labels={"ball_angle": "Arm Angle", "year": "Year"}
                )
                st.plotly_chart(fig_pitchers)
                
                # Show detailed stats for selected pitchers
                st.subheader("Detailed Statistics")
                stats_df = pitcher_data.groupby(['pitcher_name', 'year']).agg({
                    'ball_angle': 'mean',
                    'n_pitches': 'sum'
                }).round(2)
                st.dataframe(stats_df)

if __name__ == "__main__":
    main()
