import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

def load_all_seasons():
    """Load all processed seasons of data"""
    data_path = Path("data/processed")
    seasons = range(20, 25)
    dfs = []
    
    # Print loading status
    st.sidebar.write("Loading data files:")
    
    for year in seasons:
        try:
            file_path = data_path / f'ArmAngles{year}_complete.csv'
            df = pd.read_csv(file_path)
            df['year'] = f'20{year}'
            dfs.append(df)
            st.sidebar.success(f"✓ Loaded 20{year} data")
        except Exception as e:
            st.sidebar.error(f"Error loading 20{year} data: {e}")
    
    if not dfs:
        return None
        
    combined_df = pd.concat(dfs, ignore_index=True)
    st.sidebar.info(f"Total records loaded: {len(combined_df)}")
    return combined_df

def create_angle_buckets(df, bucket_size):
    """Create arm angle buckets of specified size"""
    min_angle = np.floor(df['ball_angle'].min() / bucket_size) * bucket_size
    max_angle = np.ceil(df['ball_angle'].max() / bucket_size) * bucket_size
    
    df['angle_bucket'] = pd.cut(
        df['ball_angle'],
        bins=np.arange(min_angle, max_angle + bucket_size, bucket_size),
        labels=[f"{i} to {i + bucket_size}" for i in np.arange(min_angle, max_angle, bucket_size)]
    )
    return df

def calculate_bucket_stats(df):
    """Calculate average stats for each bucket"""
    stats = [
        'k_percent', 'bb_percent', 'whiff_percent',
        'barrel_percent', 'hard_hit_percent', 'xwoba'
    ]
    
    return df.groupby('angle_bucket')[stats].agg(['mean', 'count', 'std']).round(2)

def main():
    st.title("Pitcher Arm Angle Analysis")
    
    # Load data from processed directory
    data = load_all_seasons()
    if data is None:
        st.error("No data available. Please check the data/processed directory.")
        return
    
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
    
    # Bucket size selection
    bucket_size = st.sidebar.radio(
        "Select Bucket Size",
        options=[5, 10, 15],
        help="Size of arm angle groupings in degrees"
    )
    
    # Create buckets
    data_with_buckets = create_angle_buckets(data, bucket_size)
    
    # Calculate stats by bucket
    bucket_stats = calculate_bucket_stats(data_with_buckets)
    
    # Available metrics
    metrics = {
        'K%': 'k_percent',
        'BB%': 'bb_percent',
        'Whiff%': 'whiff_percent',
        'Barrel%': 'barrel_percent',
        'Hard Hit%': 'hard_hit_percent',
        'xwOBA': 'xwoba'
    }
    
    # Create tabs for different visualization types
    tab1, tab2 = st.tabs(["Individual Metrics", "Combined View"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            # Individual metric selection
            selected_metric = st.selectbox(
                "Select Metric to Display",
                options=list(metrics.keys())
            )
        
        with col2:
            # Display type selection
            plot_type = st.selectbox(
                "Plot Type",
                options=["Box Plot", "Violin Plot", "Bar Plot"]
            )
        
        # Create visualization based on selected type
        metric_col = metrics[selected_metric]
        
        if plot_type == "Box Plot":
            fig = px.box(
                data_with_buckets,
                x='angle_bucket',
                y=metric_col,
                title=f"{selected_metric} by Arm Angle ({bucket_size}° buckets)",
                color='year' if len(selected_years) > 1 else None
            )
        elif plot_type == "Violin Plot":
            fig = px.violin(
                data_with_buckets,
                x='angle_bucket',
                y=metric_col,
                title=f"{selected_metric} by Arm Angle ({bucket_size}° buckets)",
                color='year' if len(selected_years) > 1 else None
            )
        else:  # Bar Plot
            avg_data = data_with_buckets.groupby('angle_bucket')[metric_col].mean().reset_index()
            fig = px.bar(
                avg_data,
                x='angle_bucket',
                y=metric_col,
                title=f"Average {selected_metric} by Arm Angle ({bucket_size}° buckets)"
            )
        
        fig.update_layout(
            xaxis_title="Arm Angle Range (degrees)",
            yaxis_title=selected_metric
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary statistics
        st.subheader(f"Summary Statistics for {selected_metric}")
        summary_stats = bucket_stats[metric_col].round(2)
        st.dataframe(summary_stats)
    
    with tab2:
        # Create small multiples view of all metrics
        cols = st.columns(2)
        for idx, (metric_name, metric_col) in enumerate(metrics.items()):
            fig = px.box(
                data_with_buckets,
                x='angle_bucket',
                y=metric_col,
                title=f"{metric_name} by Arm Angle",
                color='year' if len(selected_years) > 1 else None
            )
            
            fig.update_layout(
                height=400,
                xaxis_title="Arm Angle Range",
                yaxis_title=metric_name
            )
            
            cols[idx % 2].plotly_chart(fig, use_container_width=True)
        
        # Show data table with all metrics
        st.subheader("Complete Statistics by Arm Angle")
        st.dataframe(bucket_stats)

if __name__ == "__main__":
    main()
