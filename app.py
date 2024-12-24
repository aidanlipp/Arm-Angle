import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

def load_and_validate_data():
    """Load data and perform validation checks"""
    data_path = Path("data/processed")
    dfs = []
    
    # Print loading status
    st.sidebar.write("Loading and validating data:")
    
    for year in range(20, 25):
        try:
            file_path = data_path / f'ArmAngles{year}_complete.csv'
            df = pd.read_csv(file_path)
            df['year'] = f'20{year}'
            
            # Log suspicious values
            suspicious = df[df['ball_angle'].abs() > 60]
            if not suspicious.empty:
                st.sidebar.warning(f"Found {len(suspicious)} suspicious arm angles in 20{year}")
                
            dfs.append(df)
            st.sidebar.success(f"✓ Loaded 20{year} data")
        except Exception as e:
            st.sidebar.error(f"Error loading 20{year} data: {e}")
    
    if not dfs:
        return None
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Display data statistics
    st.sidebar.info(f"Arm Angle Range: {combined_df['ball_angle'].min():.1f}° to {combined_df['ball_angle'].max():.1f}°")
    
    return combined_df

def main():
    st.title("Pitcher Arm Angle Analysis")
    
    # Load and validate data
    data = load_and_validate_data()
    if data is None:
        st.error("No data available. Please check the data/processed directory.")
        return
    
    # Show data validation section
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pitchers", len(data))
    with col2:
        st.metric("Median Arm Angle", f"{data['ball_angle'].median():.1f}°")
    with col3:
        st.metric("Unusual Angles", len(data[data['ball_angle'].abs() > 60]))
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Year filter
    available_years = sorted(data['year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Years",
        options=available_years,
        default=available_years
    )
    
    # Filter by selected years
    data = data[data['year'].isin(selected_years)]
    
    # Available metrics
    metrics = {
        'K%': 'k_percent',
        'BB%': 'bb_percent',
        'Whiff%': 'whiff_percent',
        'Barrel%': 'barrel_percent',
        'Hard Hit%': 'hard_hit_percent',
        'xwOBA': 'xwoba'
    }
    
    # Metric selection
    selected_metric = st.selectbox(
        "Select Metric to Display",
        options=list(metrics.keys())
    )
    
    # Create scatter plot
    fig = px.scatter(
        data,
        x='ball_angle',
        y=metrics[selected_metric],
        color='year' if len(selected_years) > 1 else None,
        title=f"{selected_metric} vs Arm Angle",
        labels={
            'ball_angle': 'Arm Angle (degrees)',
            metrics[selected_metric]: selected_metric
        },
        trendline="lowess",  # Add smoothed trendline
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Arm Angle (degrees)",
        yaxis_title=selected_metric,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add data filters for unusual values
    st.subheader("Data Filters")
    show_unusual = st.checkbox("Show pitchers with unusual arm angles")
    if show_unusual:
        unusual_data = data[data['ball_angle'].abs() > 60].sort_values('ball_angle')
        st.dataframe(
            unusual_data[['pitcher_name', 'year', 'ball_angle'] + list(metrics.values())],
            use_container_width=True
        )
        
        st.info("These unusual values might need to be investigated for data quality issues.")

if __name__ == "__main__":
    main()
