import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

def create_angle_buckets(df, bucket_size):
    """Create arm angle buckets based on actual data range"""
    min_angle = df['ball_angle'].min()
    max_angle = df['ball_angle'].max()
    
    bucket_edges = np.arange(
        np.floor(min_angle / bucket_size) * bucket_size,
        np.ceil(max_angle / bucket_size) * bucket_size + bucket_size,
        bucket_size
    )
    
    bucket_labels = [f"{edge:.0f} to {edge + bucket_size:.0f}" 
                    for edge in bucket_edges[:-1]]
    
    df['angle_bucket'] = pd.cut(
        df['ball_angle'],
        bins=bucket_edges,
        labels=bucket_labels,
        include_lowest=True
    )
    
    return df

def load_and_validate_data():
    """Load and validate data from processed directory"""
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
            else:
                st.sidebar.warning(f"File not found: {file_path}")
        except Exception as e:
            st.sidebar.error(f"Error loading 20{year} data: {e}")
    
    if not dfs:
        return None
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Add data summary
    st.sidebar.subheader("Data Summary")
    st.sidebar.write(f"Total Records: {len(combined_df)}")
    st.sidebar.write(f"Years: {', '.join(sorted(combined_df['year'].unique()))}")
    st.sidebar.write(f"Angle Range: {combined_df['ball_angle'].min():.1f}° to {combined_df['ball_angle'].max():.1f}°")
    
    return combined_df




def get_empty_buckets(df, metric):
    """Identify buckets with no data"""
    bucket_counts = df.groupby(['year', 'angle_bucket']).size().unstack(fill_value=0)
    empty_buckets = bucket_counts.columns[bucket_counts.sum() == 0]
    return empty_buckets

def create_visualization(data, metric_name, metric_col, plot_type, bucket_size=None):
    """Create visualization with league average line"""
    
    if plot_type == "Scatter":
        fig = px.scatter(
            data,
            x='ball_angle',
            y=metric_col,
            color='year',
            title=f"{metric_name} vs Arm Angle"
        )
        
        # Add league average line
        league_avg = data[metric_col].mean()
        fig.add_hline(
            y=league_avg,
            line_dash="dash",
            line_color="red",
            annotation_text=f"League Avg: {league_avg:.1f}",
            annotation_position="bottom right"
        )
        
    else:
        data_with_buckets = create_angle_buckets(data.copy(), bucket_size)
        
        # Remove empty buckets and note them
        empty_buckets = get_empty_buckets(data_with_buckets, metric_col)
        if not empty_buckets.empty:
            empty_bucket_note = "Empty buckets: " + ", ".join(empty_buckets)
            data_with_buckets = data_with_buckets[~data_with_buckets['angle_bucket'].isin(empty_buckets)]
        
        if plot_type == "Box Plot":
            fig = px.box(
                data_with_buckets,
                x='angle_bucket',
                y=metric_col,
                color='year',
                title=f"{metric_name} by Arm Angle ({bucket_size}° buckets)"
            )
        else:  # Bar Chart
            avg_data = data_with_buckets.groupby('angle_bucket')[metric_col].agg([
                'mean', 'count'
            ]).reset_index()
            
            fig = px.bar(
                avg_data,
                x='angle_bucket',
                y='mean',
                title=f"Average {metric_name} by Arm Angle ({bucket_size}° buckets)",
                text=avg_data['count'].apply(lambda x: f"n={x}")  # Add sample size labels
            )
            
            # Optimize y-axis range
            y_min = avg_data['mean'].min() * 0.95
            y_max = avg_data['mean'].max() * 1.05
            fig.update_layout(yaxis_range=[y_min, y_max])
        
        # Add league average line
        league_avg = data[metric_col].mean()
        fig.add_hline(
            y=league_avg,
            line_dash="dash",
            line_color="red",
            annotation_text=f"League Avg: {league_avg:.1f}",
            annotation_position="bottom right"
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Arm Angle (degrees)",
        yaxis_title=metric_name,
        showlegend=True
    )
    
    return fig, empty_buckets if 'empty_buckets' in locals() else None

def main():
    st.title("Pitcher Arm Angle Analysis")
    
    # Load data
    data = load_and_validate_data()  # Your existing load function
    if data is None:
        st.error("No data available. Please check the data/processed directory.")
        return
    
    # Available metrics with typical ranges for y-axis optimization
    metrics = {
        'K%': {'col': 'k_percent', 'range': [0, 45]},
        'BB%': {'col': 'bb_percent', 'range': [0, 20]},
        'Whiff%': {'col': 'whiff_percent', 'range': [0, 40]},
        'Barrel%': {'col': 'barrel_percent', 'range': [0, 15]},
        'Hard Hit%': {'col': 'hard_hit_percent', 'range': [20, 50]},
        'xwOBA': {'col': 'xwoba', 'range': [0.250, 0.400]}
    }
    
    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_metric = st.selectbox("Select Metric", options=list(metrics.keys()))
    with col2:
        plot_type = st.selectbox("Plot Type", ["Bar Chart", "Scatter", "Box Plot"])
    with col3:
        if plot_type != "Scatter":
            bucket_size = st.selectbox("Bucket Size", [5, 10, 15], index=1)
    
    # Create visualization
    fig, empty_buckets = create_visualization(
        data,
        selected_metric,
        metrics[selected_metric]['col'],
        plot_type,
        bucket_size if plot_type != "Scatter" else None
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display empty bucket information if any
    if empty_buckets is not None and not empty_buckets.empty:
        st.info(f"The following angle ranges had no data: {', '.join(empty_buckets)}")
    
    # Add sample size information
    if plot_type != "Scatter":
        st.caption("Note: 'n=' values show the number of pitchers in each bucket")

if __name__ == "__main__":
    main()
