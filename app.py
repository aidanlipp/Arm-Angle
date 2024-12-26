import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from streamlit_plotly_events import plotly_events

def load_and_validate_data():
    data_path = Path("data/processed")
    dfs = []
    
    for year in range(20, 24):  # Update range if you have 2024 data
        file_path = data_path / f'ArmAngles{year}_complete.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['year'] = f'20{year}'
            dfs.append(df)
            st.sidebar.success(f"✓ Loaded 20{year} data")
        else:
            st.sidebar.warning(f"Missing: ArmAngles{year}_complete.csv")
    
    if not dfs:
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def create_angle_buckets(df, bucket_size):
    min_angle = np.floor(df['ball_angle'].min() / bucket_size) * bucket_size
    max_angle = np.ceil(df['ball_angle'].max() / bucket_size) * bucket_size
    
    bins = np.arange(min_angle, max_angle + bucket_size, bucket_size)
    labels = [f"{bins[i]:.0f} to {bins[i+1]:.0f}" for i in range(len(bins)-1)]
    
    df['angle_bucket'] = pd.cut(df['ball_angle'], bins=bins, labels=labels)
    return df

def main():
    st.title("Pitcher Arm Angle Analysis")
    
    data = load_and_validate_data()
    if data is None:
        st.error("No data available")
        return

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
    
    if plot_type == "Bar Chart":
        data_with_buckets = create_angle_buckets(data.copy(), bucket_size)
        bucket_stats = data_with_buckets.groupby('angle_bucket').agg({
            metrics[selected_metric]: ['mean', 'count'],
            'pitcher_name': list
        }).reset_index()
        
        fig = px.bar(
            bucket_stats,
            x='angle_bucket',
            y=(metrics[selected_metric], 'mean'),
            text=bucket_stats[(metrics[selected_metric], 'count')].apply(lambda x: f"n={x}"),
            title=f"Average {selected_metric} by Arm Angle ({bucket_size}° buckets)"
        )
        
        league_avg = data[metrics[selected_metric]].mean()
        fig.add_hline(
            y=league_avg,
            line_dash="dash",
            line_color="red",
            annotation_text=f"League Avg: {league_avg:.1f}"
        )
    else:
        fig = px.scatter(
            data,
            x='ball_angle',
            y=metrics[selected_metric],
            title=f"{selected_metric} vs Arm Angle",
            hover_data=['pitcher_name', 'year']
        )

    selected_point = plotly_events(fig, click_event=True)
    
    if selected_point:
        st.subheader("Selected Pitcher(s)")
        if plot_type == "Bar Chart":
            bucket = selected_point[0]['x']
            selected_pitchers = data_with_buckets[data_with_buckets['angle_bucket'] == bucket]
        else:
            clicked_x = selected_point[0]['x']
            clicked_y = selected_point[0]['y']
            selected_pitchers = data[
                (abs(data['ball_angle'] - clicked_x) < 0.5) &
                (abs(data[metrics[selected_metric]] - clicked_y) < 0.5)
            ]
        
        if not selected_pitchers.empty:
            st.dataframe(
                selected_pitchers[['pitcher_name', 'year', 'ball_angle'] + list(metrics.values())]
                .sort_values('ball_angle')
                .round(2)
            )

if __name__ == "__main__":
    main()
