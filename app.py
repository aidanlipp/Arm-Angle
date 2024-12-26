import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from streamlit_plotly_events import plotly_events

def load_and_validate_data():
    """Load data from processed directory"""
    data_path = Path("data/processed")
    dfs = []
    
    for year in range(20, 25):
        try:
            file_path = data_path / f'ArmAngles{year}_complete.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['year'] = f'20{year}'
                dfs.append(df)
                st.sidebar.success(f"✓ Loaded 20{year} data")
        except Exception as e:
            st.sidebar.warning(f"Missing file: ArmAngles{year}_complete.csv")
    
    if not dfs:
        return None
        
    combined_df = pd.concat(dfs, ignore_index=True)
    st.sidebar.write(f"Years: {', '.join(sorted(combined_df['year'].unique()))}")
    st.sidebar.write(f"Angle Range: {combined_df['ball_angle'].min():.1f}° to {combined_df['ball_angle'].max():.1f}°")
    
    return combined_df

def create_angle_buckets(df, bucket_size):
    min_angle = np.floor(df['ball_angle'].min() / bucket_size) * bucket_size
    max_angle = np.ceil(df['ball_angle'].max() / bucket_size) * bucket_size
    
    df['angle_bucket'] = pd.cut(
        df['ball_angle'],
        bins=np.arange(min_angle, max_angle + bucket_size, bucket_size),
        labels=[f"{i} to {i + bucket_size}" for i in np.arange(min_angle, max_angle, bucket_size)]
    )
    return df

def main():
    st.title("Pitcher Arm Angle Analysis")
    
    data = load_and_validate_data()
    if data is None:
        st.error("No data available")
        return
    
    # Available metrics
    metrics = {
        'K%': 'k_percent',
        'BB%': 'bb_percent',
        'Whiff%': 'whiff_percent',
        'Barrel%': 'barrel_percent',
        'Hard Hit%': 'hard_hit_percent',
        'xwOBA': 'xwoba'
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_metric = st.selectbox("Select Metric", list(metrics.keys()))
    with col2:
        plot_type = st.selectbox("Plot Type", ["Bar Chart", "Scatter"])
    with col3:
        bucket_size = st.selectbox("Bucket Size", [5, 10, 15]) if plot_type == "Bar Chart" else None
    
    metric_col = metrics[selected_metric]
    
    if plot_type == "Bar Chart":
        data_with_buckets = create_angle_buckets(data.copy(), bucket_size)
        
        # Calculate averages and counts per bucket
        bucket_stats = data_with_buckets.groupby('angle_bucket').agg({
            metric_col: ['mean', 'count']
        }).reset_index()
        
        fig = px.bar(
            bucket_stats,
            x='angle_bucket',
            y=('metric_col', 'mean'),
            text=bucket_stats[('metric_col', 'count')].apply(lambda x: f"n={x}"),
            title=f"Average {selected_metric} by Arm Angle ({bucket_size}° buckets)"
        )
        
        # Add league average line
        league_avg = data[metric_col].mean()
        fig.add_hline(y=league_avg, line_dash="dash", line_color="red",
                     annotation_text=f"League Avg: {league_avg:.1f}")
        
    else:  # Scatter plot
        fig = px.scatter(
            data,
            x='ball_angle',
            y=metric_col,
            title=f"{selected_metric} vs Arm Angle",
            hover_data=['pitcher_name', 'year']
        )
    
    # Show plot and capture clicks
    selected_point = plotly_events(fig)
    
    # Show selected data
    if selected_point:
        st.subheader("Selected Pitcher(s)")
        if plot_type == "Bar Chart":
            bucket = selected_point[0]['x']
            bucket_data = data_with_buckets[data_with_buckets['angle_bucket'] == bucket]
            st.dataframe(
                bucket_data[['pitcher_name', 'year', 'ball_angle'] + list(metrics.values())]
                .sort_values('ball_angle')
            )
        else:
            clicked_angle = selected_point[0]['x']
            clicked_metric = selected_point[0]['y']
            closest_pitcher = data[
                (abs(data['ball_angle'] - clicked_angle) < 0.1) &
                (abs(data[metric_col] - clicked_metric) < 0.1)
            ]
            st.dataframe(
                closest_pitcher[['pitcher_name', 'year', 'ball_angle'] + list(metrics.values())]
            )

if __name__ == "__main__":
    main()
