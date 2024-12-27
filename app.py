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
    
    for year in range(20, 24):
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
    
    # Year selection
    available_years = sorted(data['year'].unique())
    selected_years = st.multiselect(
        "Select Years",
        options=available_years,
        default=available_years,
        key='year_selector'
    )
    
    if selected_years:
        data = data[data['year'].isin(selected_years)]
    else:
        st.warning("Please select at least one year")
        return
    
if plot_type == "Scatter":
        fig = create_scatter_plot(data, selected_metric, metrics)   
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
    
    # Display plot and handle clicks
    selected_point = plotly_events(fig, click_event=True)
    
    if selected_point:
        st.subheader("Selected Pitcher(s)")
        if plot_type == "Bar Chart":
            bucket = selected_point[0]['x']
            selected_data = data_with_buckets[data_with_buckets['angle_bucket'] == bucket]
        else:
            clicked_x = selected_point[0]['x']
            clicked_y = selected_point[0]['y']
            selected_data = data[
                (abs(data['ball_angle'] - clicked_x) < 0.5) &
                (abs(data[metrics[selected_metric]] - clicked_y) < 0.5)
            ]
        
        if not selected_data.empty:
            display_cols = ['pitcher_name', 'year', 'ball_angle'] + list(metrics.values())
            st.dataframe(
                selected_data[display_cols]
                .sort_values('ball_angle')
                .round(3)
            )

if __name__ == "__main__":
    main()
