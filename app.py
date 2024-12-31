import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from streamlit_plotly_events import plotly_events

# Keep overall league averages
overall_league_averages = {
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

def load_and_validate_data():
    """Load and validate data from processed directory"""
    data_path = Path("data/processed")
    dfs = []
    
    for year in ['2020', '2021', '2022', '2023', '2024']:
        short_year = year[-2:]
        file_path = data_path / f'ArmAngles{short_year}_complete.csv'
        try:
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['year'] = year
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

def load_specific_averages():
    """Load specific league averages from CSV files"""
    data_path = Path("data/league_averages")
    specific_averages = {}
    
    for category in ['RHStarters', 'LHStarters', 'RHRelievers', 'LHRelievers']:
        file_path = data_path / f'LeagueAvg{category}.csv'
        try:
            if file_path.exists():
                df = pd.read_csv(file_path)
                yearly_stats = {}
                for metric in ['k_percent', 'bb_percent', 'barrel_percent', 'hard_hit_percent', 'xwoba']:
                    yearly_stats[metric] = df.groupby('year')[metric].mean().to_dict()
                specific_averages[category] = yearly_stats
                st.sidebar.success(f"✓ Loaded {category} averages")
            else:
                st.sidebar.warning(f"Missing: {file_path}")
        except Exception as e:
            st.sidebar.error(f"Error loading {category} averages: {e}")
    
    return specific_averages

def get_appropriate_average(metric, years, specific_averages=None, filters=None):
    """Get appropriate league average based on filters"""
    if not filters or not filters.get('handedness') or not filters.get('role'):
        # Use overall average if no specific filters
        valid_years = [year for year in years if year in overall_league_averages[metric]]
        if not valid_years:
            return None
        return sum(overall_league_averages[metric][year] for year in valid_years) / len(valid_years)
    
    # Use specific average if available and filters match
    if specific_averages:
        key = f"{filters['handedness']}{'Starters' if filters['role'] == 'Starter' else 'Relievers'}"
        if key in specific_averages and metric in specific_averages[key]:
            stats = specific_averages[key][metric]
            valid_years = [year for year in years if str(year) in stats]
            if valid_years:
                return sum(stats[str(year)] for year in valid_years) / len(valid_years)
    
    # Fall back to overall average
    return get_appropriate_average(metric, years)

def create_visualization(data, selected_metric, specific_averages, bucket_size=None, plot_type="Bar Chart"):
    """Create visualization with appropriate league averages"""
    metrics = {
        'K%': 'k_percent',
        'BB%': 'bb_percent',
        'Whiff%': 'whiff_percent',
        'Barrel%': 'barrel_percent',
        'Hard Hit%': 'hard_hit_percent',
        'xwOBA': 'xwoba'
    }
    
    metric_key = metrics[selected_metric]
    
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
    else:
        fig = px.scatter(
            data,
            x='ball_angle',
            y=metrics[selected_metric],
            color='year',
            title=f"{selected_metric} vs Arm Angle",
            hover_data=['pitcher_name', 'year', 'pitch_hand', 'role'],
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        y_min = data[metrics[selected_metric]].min() * 0.98
        y_max = data[metrics[selected_metric]].max() * 1.02
    
    # Add league average lines
    years = list(data['year'].unique())
    
    # Get unique combinations in the filtered data
    combinations = data.groupby(['pitch_hand', 'role']).size().reset_index()[['pitch_hand', 'role']]
    
    if len(combinations) == 1:
        # If only one combination is selected, show specific average
        filters = {'handedness': combinations.iloc[0]['pitch_hand'], 
                  'role': combinations.iloc[0]['role']}
        avg = get_appropriate_average(metric_key, years, specific_averages, filters)
        if avg is not None:
            fig.add_hline(
                y=avg,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{filters['handedness']}{filters['role']} Avg: {avg:.3f}" if metric_key == 'xwoba' else f"{filters['handedness']}{filters['role']} Avg: {avg:.1f}",
                annotation_position="bottom right"
            )
    else:
        # Show overall league average if multiple combinations are selected
        avg = get_appropriate_average(metric_key, years)
        if avg is not None:
            fig.add_hline(
                y=avg,
                line_dash="dash",
                line_color="red",
                annotation_text=f"League Avg: {avg:.3f}" if metric_key == 'xwoba' else f"League Avg: {avg:.1f}",
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
    
    # Load both types of data
    data = load_and_validate_data()
    specific_averages = load_specific_averages()
    
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
    st.subheader("Select Years")
    available_years = sorted(data['year'].unique())
    year_cols = st.columns(len(available_years))
    selected_years = []
    
    for idx, year in enumerate(available_years):
        with year_cols[idx]:
            if st.checkbox(year, value=True, key=f'year_{year}'):
                selected_years.append(year)
    
    if not selected_years:
        st.warning("Please select at least one year")
        return
    
    filtered_data = data[data['year'].isin(selected_years)]
    
    # Add pitcher handedness filter
    st.subheader("Filter by Pitcher Handedness")
    handedness_options = ['L', 'R']
    selected_handedness = st.multiselect("Select Pitch Hand", handedness_options, default=handedness_options)
    if selected_handedness:
        filtered_data = filtered_data[filtered_data['pitch_hand'].isin(selected_handedness)]
    
    # Add role filter
    st.subheader("Filter by Role")
    role_options = ['Starter', 'Reliever']
    selected_roles = st.multiselect("Select Role", role_options, default=role_options)
    if selected_roles:
        filtered_data = filtered_data[filtered_data['role'].isin(selected_roles)]
    
    # Create visualization with both types of averages
    fig = create_visualization(
        filtered_data, 
        selected_metric,
        specific_averages,
        bucket_size=bucket_size, 
        plot_type=plot_type
    )
    
    # Display plot and handle clicks
    selected_point = plotly_events(fig, click_event=True)
    
    if selected_point:
        st.subheader("Selected Pitcher(s)")
        if plot_type == "Bar Chart":
            bucket = selected_point[0]['x']
            data_with_buckets = create_angle_buckets(filtered_data.copy(), bucket_size)
            selected_data = data_with_buckets[data_with_buckets['angle_bucket'] == bucket]
        else:
            clicked_x = selected_point[0]['x']
            clicked_y = selected_point[0]['y']
            selection_radius = 1.0
            selected_data = filtered_data[
                (abs(filtered_data['ball_angle'] - clicked_x) < selection_radius) & 
                (abs(filtered_data[metrics[selected_metric]] - clicked_y) < selection_radius)
            ]
        
        if not selected_data.empty:
            display_cols = ['pitcher_name', 'year', 'ball_angle', 'pitch_hand', 'role'] + list(metrics.values())
            st.dataframe(
                selected_data[display_cols]
                .sort_values(['year', 'ball_angle'])
                .round(3)
            )
            
            st.caption(f"Found {len(selected_data)} pitcher(s) in this selection")

if __name__ == "__main__":
    main()
