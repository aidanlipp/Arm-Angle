import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from streamlit_plotly_events import plotly_events

def load_and_validate_data():
    """Load and validate data from processed directory"""
    data_path = Path("data/processed")
    dfs = []
    
    # Explicitly try to load each year
    for year in ['2020', '2021', '2022', '2023', '2024']:
        short_year = year[-2:]  # Get last two digits
        file_path = data_path / f'ArmAngles{short_year}_complete.csv'
        try:
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['year'] = year  # Use full year string
                dfs.append(df)
                st.sidebar.success(f"✓ Loaded {year} data")
            else:
                st.sidebar.warning(f"Missing: ArmAngles{short_year}_complete.csv")
        except Exception as e:
            st.sidebar.error(f"Error loading {year} data: {e}")

    if not dfs:
        return None

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def create_angle_buckets(df, bucket_size):
    """Create angle buckets with proper error handling"""
    if df.empty:
        return df
        
    min_angle = df['ball_angle'].min()
    max_angle = df['ball_angle'].max()
    
    min_edge = np.floor(min_angle / bucket_size) * bucket_size
    max_edge = np.ceil(max_angle / bucket_size) * bucket_size
    edges = np.arange(min_edge, max_edge + bucket_size, bucket_size)
    
    labels = [f"{edges[i]:.0f} to {edges[i+1]:.0f}" for i in range(len(edges)-1)]
    
    df['angle_bucket'] = pd.cut(
        df['ball_angle'],
        bins=edges,
        labels=labels,
        include_lowest=True
    )
    
    return df

def create_visualization(data, selected_metric, bucket_size=None, plot_type="Bar Chart"):
    """Create visualization with official league averages"""
    league_averages = {
        'k_percent': {
            '2020': 23.4, '2021': 23.2, '2022': 22.4, '2023': 22.7, '2024': 22.6
        },
        'bb_percent': {
            '2020': 9.2, '2021': 8.7, '2022': 8.2, '2023': 8.6, '2024': 8.2
        },
        'barrel_percent': {
            '2020': 7.6, '2021': 7.9, '2022': 7.5, '2023': 8.1, '2024': 7.8
        },
        'hard_hit_percent': {
            '2020': 37.4, '2021': 38.5, '2022': 38.2, '2023': 39.2, '2024': 38.7
        },
        'xwoba': {
            '2020': .323, '2021': .317, '2022': .309, '2023': .320, '2024': .312
        }
    }
    
    metrics = {
        'K%': 'k_percent',
        'BB%': 'bb_percent',
        'Whiff%': 'whiff_percent',
        'Barrel%': 'barrel_percent',
        'Hard Hit%': 'hard_hit_percent',
        'xwOBA': 'xwoba'
    }
    
    if plot_type == "Bar Chart":
        data_with_buckets = create_angle_buckets(data.copy(), bucket_size)
        bucket_stats = data_with_buckets.groupby('angle_bucket', observed=True).agg({
            metrics[selected_metric]: ['mean', 'count']
        }).reset_index()
        bucket_stats.columns = ['angle_bucket', 'mean', 'count']
        
        fig = px.bar(
            bucket_stats,
            x='angle_bucket',
            y='mean',
            title=f"Average {selected_metric} by Arm Angle ({bucket_size}° buckets)"
        )
        fig.update_traces(marker_color='rgb(0, 116, 217)', showlegend=False)
        
        y_min = bucket_stats['mean'].min() * 0.98
        y_max = bucket_stats['mean'].max() * 1.02
        
    else:  # Scatter Plot
        fig = px.scatter(
            data,
            x='ball_angle',
            y=metrics[selected_metric],
            color='year',
            title=f"{selected_metric} vs Arm Angle",
            hover_data=['pitcher_name', 'year'],
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        y_min = data[metrics[selected_metric]].min() * 0.98
        y_max = data[metrics[selected_metric]].max() * 1.02
    
    # Get relevant league average based on selected years
    years = list(data['year'].unique())
    metric_key = metrics[selected_metric]
    metric_averages = [league_averages[metric_key][year] for year in years]
    league_avg = sum(metric_averages) / len(metric_averages)
    fig.add_hline(
        y=league_avg,
        line_dash="dash",
        line_color="red",
        annotation_text=f"League Avg: {league_avg:.3f}" if metric_key == 'xwoba' else f"League Avg: {league_avg:.1f}",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        xaxis_title="Arm Angle (degrees)",
        yaxis_title=selected_metric,
        yaxis=dict(
            range=[y_min, y_max],
            tickformat='.3f' if metric_key == 'xwoba' else '.1f'
        ),
        plot_bgcolor='white'
    )
    
    return fig

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
    
    # Year selection - now using checkboxes for easier toggling
    st.subheader("Select Years")
    available_years = sorted(data['year'].unique())
    
    # Create columns for year checkboxes
    year_cols = st.columns(len(available_years))
    selected_years = []
    
    for idx, year in enumerate(available_years):
        with year_cols[idx]:
            if st.checkbox(year, value=True, key=f'year_{year}'):
                selected_years.append(year)
    
    if not selected_years:
        st.warning("Please select at least one year")
        return
    
    data = data[data['year'].isin(selected_years)]
    
    # Use the new create_visualization function
    fig = create_visualization(
        data, 
        selected_metric, 
        bucket_size=bucket_size, 
        plot_type=plot_type
    )
    
    # Display plot and handle clicks with increased selection radius
    selected_point = plotly_events(fig, click_event=True)
    
    if selected_point:
        st.subheader("Selected Pitcher(s)")
        if plot_type == "Bar Chart":
            bucket = selected_point[0]['x']
            if 'data_with_buckets' not in locals():
                data_with_buckets = create_angle_buckets(data.copy(), bucket_size)
            selected_data = data_with_buckets[data_with_buckets['angle_bucket'] == bucket]
        else:
            clicked_x = selected_point[0]['x']
            clicked_y = selected_point[0]['y']
            # Increased selection radius for easier point selection
            selection_radius = 1.0  # Increased from 0.5
            selected_data = data[
                (abs(data['ball_angle'] - clicked_x) < selection_radius) &
                (abs(data[metrics[selected_metric]] - clicked_y) < selection_radius)
            ]
        
        if not selected_data.empty:
            display_cols = ['pitcher_name', 'year', 'ball_angle'] + list(metrics.values())
            st.dataframe(
                selected_data[display_cols]
                .sort_values(['year', 'ball_angle'])
                .round(3)
            )
            
            # Show how many pitchers were selected
            st.caption(f"Found {len(selected_data)} pitcher(s) in this selection")

if __name__ == "__main__":
    main()
