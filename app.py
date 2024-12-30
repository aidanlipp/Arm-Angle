import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from streamlit_plotly_events import plotly_events
import requests
from bs4 import BeautifulSoup
from datetime import datetime

def validate_league_stats(data):
    """Compare stats with FanGraphs league averages"""
    league_stats = {
        '2020': {'K%': 23.4, 'BB%': 9.2, 'Barrel%': 7.6, 'HardHit%': 37.4},
        '2021': {'K%': 23.2, 'BB%': 8.7, 'Barrel%': 7.9, 'HardHit%': 38.5},
        '2022': {'K%': 22.4, 'BB%': 8.2, 'Barrel%': 7.5, 'HardHit%': 38.2},
        '2023': {'K%': 22.7, 'BB%': 8.6, 'Barrel%': 8.1, 'HardHit%': 39.2},
        '2024': {'K%': 22.6, 'BB%': 8.2, 'Barrel%': 7.8, 'HardHit%': 38.7}
    }
    
    st.subheader("Stats Validation Against FanGraphs")
    
    metrics = {
        'K%': 'k_percent',
        'BB%': 'bb_percent',
        'Barrel%': 'barrel_percent',
        'HardHit%': 'hard_hit_percent'
    }
    
    comparison = []
    for year in sorted(data['year'].unique()):
        row = {'Year': year}
        for metric, our_col in metrics.items():
            our_val = data[data['year'] == year][our_col].mean()
            fg_val = league_stats[year][metric]
            row.update({
                f'Our {metric}': round(our_val, 1),
                f'FG {metric}': fg_val,
                f'{metric} Diff': round(abs(our_val - fg_val), 1)
            })
        comparison.append(row)
    
    df_comparison = pd.DataFrame(comparison)
    
    # Display comparison
    st.dataframe(df_comparison.style.highlight_grid())
    
    # Flag significant differences
    for metric in metrics.keys():
        large_diffs = df_comparison[df_comparison[f'{metric} Diff'] > 1.0]
        if not large_diffs.empty:
            st.warning(f"Notable differences in {metric}:")
            for _, row in large_diffs.iterrows():
                st.write(f"{row['Year']}: Our {row[f'Our {metric}']}% vs FanGraphs {row[f'FG {metric}']}%")

# Add to main():
if st.checkbox("Validate Stats"):
    validate_league_stats(data)

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

def create_scatter_plot(data, selected_metric, metrics):
    """Create scatter plot with colored points by year and metric average line"""
    # Create scatter plot with different colors by year
    fig = px.scatter(
        data,
        x='ball_angle',
        y=metrics[selected_metric],
        color='year',  # Always color by year
        title=f"{selected_metric} vs Arm Angle",
        hover_data=['pitcher_name', 'year'],
        color_discrete_sequence=px.colors.qualitative.Set1  # Use a nice color sequence
    )
    
    # Add metric average line
    metric_avg = data[metrics[selected_metric]].mean()
    fig.add_hline(
        y=metric_avg,
        line_dash="dash",
        line_color="red",
        annotation_text=f"League Avg {selected_metric}: {metric_avg:.3f}",
        annotation_position="bottom right"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Arm Angle (degrees)",
        yaxis_title=selected_metric,
        showlegend=True,
        legend_title="Year"
    )
    
    return fig

def create_bar_chart(bucket_stats, selected_metric, bucket_size, data, metrics):
    """Create bar chart with optimized visualization"""
    y_min = bucket_stats['mean'].min() * 0.98
    y_max = bucket_stats['mean'].max() * 1.02
    
    fig = px.bar(
        bucket_stats,
        x='angle_bucket',
        y='mean',
        title=f"Average {selected_metric} by Arm Angle ({bucket_size}° buckets)"
    )
    
    fig.update_traces(
        marker_color='rgb(0, 116, 217)',
        showlegend=False
    )
    
    league_avg = data[metrics[selected_metric]].mean()
    fig.add_hline(
        y=league_avg,
        line_dash="dash",
        line_color="red",
        annotation_text=f"League Avg: {league_avg:.3f}",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        xaxis_title="Arm Angle (degrees)",
        yaxis_title=selected_metric,
        yaxis=dict(
            range=[y_min, y_max],
            tickformat='.3f'
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
    
    if plot_type == "Scatter":
        fig = px.scatter(
            data,
            x='ball_angle',
            y=metrics[selected_metric],
            color='year',
            title=f"{selected_metric} vs Arm Angle",
            hover_data=['pitcher_name', 'year'],
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        metric_avg = data[metrics[selected_metric]].mean()
        fig.add_hline(
            y=metric_avg,
            line_dash="dash",
            line_color="red",
            annotation_text=f"League Avg {selected_metric}: {metric_avg:.3f}",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            xaxis_title="Arm Angle (degrees)",
            yaxis_title=selected_metric,
            showlegend=True,
            legend_title="Year"
        )
            
    else:  # Bar Chart
        try:
            data_with_buckets = create_angle_buckets(data.copy(), bucket_size)
            bucket_stats = data_with_buckets.groupby('angle_bucket', observed=True).agg({
                metrics[selected_metric]: ['mean', 'count']
            }).reset_index()
            bucket_stats.columns = ['angle_bucket', 'mean', 'count']
            fig = create_bar_chart(bucket_stats, selected_metric, bucket_size, data, metrics)
        except Exception as e:
            st.error(f"Error creating bar chart: {str(e)}")
            return
    
    # Display plot and handle clicks with increased selection radius
    selected_point = plotly_events(fig, click_event=True)
    
    if selected_point:
        st.subheader("Selected Pitcher(s)")
        if plot_type == "Bar Chart":
            bucket = selected_point[0]['x']
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
