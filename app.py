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
            suspicious = df[df['ball_angle'] < 0]  # Only flag negative angles
            if not suspicious.empty:
                st.sidebar.warning(f"Found {len(suspicious)} negative arm angles in 20{year}")
                
            dfs.append(df)
            st.sidebar.success(f"✓ Loaded 20{year} data")
        except Exception as e:
            st.sidebar.error(f"Error loading 20{year} data: {e}")
    
    if not dfs:
        return None
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Display actual data range
    min_angle = combined_df['ball_angle'].min()
    max_angle = combined_df['ball_angle'].max()
    st.sidebar.info(f"Actual Arm Angle Range: {min_angle:.1f}° to {max_angle:.1f}°")
    
    return combined_df

def create_angle_buckets(df, bucket_size):
    """Create arm angle buckets based on actual data range"""
    # Get actual min and max values
    min_angle = df['ball_angle'].min()
    max_angle = df['ball_angle'].max()
    
    # Create bucket edges based on actual data
    bucket_edges = np.arange(
        np.floor(min_angle / bucket_size) * bucket_size,
        np.ceil(max_angle / bucket_size) * bucket_size + bucket_size,
        bucket_size
    )
    
    # Create bucket labels
    bucket_labels = [f"{edge:.0f} to {edge + bucket_size:.0f}" 
                    for edge in bucket_edges[:-1]]
    
    # Add bucket column
    df['angle_bucket'] = pd.cut(
        df['ball_angle'],
        bins=bucket_edges,
        labels=bucket_labels,
        include_lowest=True
    )
    
    return df

def main():
    st.title("Pitcher Arm Angle Analysis")
    
    # Load data
    data = load_and_validate_data()
    if data is None:
        st.error("No data available. Please check the data/processed directory.")
        return
    
    # Show data validation
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pitchers", len(data))
    with col2:
        st.metric("Median Arm Angle", f"{data['ball_angle'].median():.1f}°")
    with col3:
        st.metric("Negative Angles", len(data[data['ball_angle'] < 0]))
    
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
    
    # Metrics
    metrics = {
        'K%': 'k_percent',
        'BB%': 'bb_percent',
        'Whiff%': 'whiff_percent',
        'Barrel%': 'barrel_percent',
        'Hard Hit%': 'hard_hit_percent',
        'xwOBA': 'xwoba'
    }
    
    # Plot controls
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_metric = st.selectbox(
            "Select Metric",
            options=list(metrics.keys())
        )
    with col2:
        plot_type = st.selectbox(
            "Plot Type",
            options=["Scatter", "Box Plot", "Bar Chart"]
        )
    
    # Create visualization
    if plot_type == "Scatter":
        fig = px.scatter(
            data,
            x='ball_angle',
            y=metrics[selected_metric],
            color='year' if len(selected_years) > 1 else None,
            title=f"{selected_metric} vs Arm Angle",
            labels={
                'ball_angle': 'Arm Angle (degrees)',
                metrics[selected_metric]: selected_metric
            }
        )
    else:
        # Create buckets for box plot or bar chart
        bucket_size = st.slider("Bucket Size (degrees)", 5, 15, 10, 5)
        data_with_buckets = create_angle_buckets(data, bucket_size)
        
        if plot_type == "Box Plot":
            fig = px.box(
                data_with_buckets,
                x='angle_bucket',
                y=metrics[selected_metric],
                color='year' if len(selected_years) > 1 else None,
                title=f"{selected_metric} by Arm Angle ({bucket_size}° buckets)"
            )
        else:  # Bar Chart
            avg_data = data_with_buckets.groupby('angle_bucket')[metrics[selected_metric]].mean().reset_index()
            fig = px.bar(
                avg_data,
                x='angle_bucket',
                y=metrics[selected_metric],
                title=f"Average {selected_metric} by Arm Angle ({bucket_size}° buckets)"
            )
    
    fig.update_layout(
        xaxis_title="Arm Angle (degrees)",
        yaxis_title=selected_metric
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show data table with suspicious values
    if st.checkbox("Show pitchers with negative arm angles"):
        suspicious_data = data[data['ball_angle'] < 0].sort_values('ball_angle')
        st.dataframe(
            suspicious_data[['pitcher_name', 'year', 'ball_angle'] + list(metrics.values())],
            use_container_width=True
        )

if __name__ == "__main__":
    main()
