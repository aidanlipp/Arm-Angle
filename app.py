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
    league_avg_files = {
        'RH_Starters': 'LeagueAvgRHStarters.csv',
        'LH_Starters': 'LeagueAvgLHStarters.csv',
        'RH_Relievers': 'LeagueAvgRHRelievers.csv',
        'LH_Relievers': 'LeagueAvgLHRelievers.csv'
    }
    
   def load_and_validate_data():
    """Load and validate data from processed directory"""
    data_path = Path("data/processed")
    league_avg_files = {
        'RH_Starters': 'LeagueAvgRHStarters.csv',
        'LH_Starters': 'LeagueAvgLHStarters.csv',
        'RH_Relievers': 'LeagueAvgRHRelievers.csv',
        'LH_Relievers': 'LeagueAvgLHRelievers.csv'
    }
    
    # Load player-level data
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

    # Combine player-level data
    player_data = pd.concat(dfs, ignore_index=True) if dfs else None

    # Load league average data
    league_data = {}
    for key, file_name in league_avg_files.items():
        try:
            file_path = data_path / file_name
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Rename 'swing_miss_percent' to 'whiff_percent'
                if 'swing_miss_percent' in df.columns:
                    df.rename(columns={'swing_miss_percent': 'whiff_percent'}, inplace=True)
                league_data[key] = df
                st.sidebar.success(f"✓ Loaded {key} league averages")
                st.write(f"Columns for {key}: {df.columns.tolist()}")
            else:
                st.sidebar.warning(f"Missing: {file_name}")
        except Exception as e:
            st.sidebar.error(f"Error loading {key} league averages: {e}")

    return player_data, league_data




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

def create_visualization(data, league_data, selected_metric, bucket_size=None, plot_type="Bar Chart", handedness=None, role=None):
    """Create visualization with accurate league averages"""
    metrics = {
        'K%': 'k_percent',
        'BB%': 'bb_percent',
        'Whiff%': 'whiff_percent',
        'Barrel%': 'barrel_percent',
        'Hard Hit%': 'hard_hit_percent',
        'xwOBA': 'xwoba'
    }
    
    if handedness and role:
        league_key = f"{handedness}_{role}s"
        if league_key not in league_data:
            st.error("League average data for this filter is unavailable.")
            return None
        league_averages = league_data[league_key]
    else:
        st.warning("Filters must include handedness and role.")
        return None

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
    
    # Add league average line
    metric_key = metrics[selected_metric]
    for year in sorted(data['year'].unique()):
        league_avg = league_averages[league_averages['year'] == int(year)][metric_key].values[0]
        fig.add_hline(
            y=league_avg,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{year} League Avg: {league_avg:.3f}" if metric_key == 'xwoba' else f"{year} League Avg: {league_avg:.1f}",
            annotation_position="bottom right"
        )
    
    return fig
    
def create_league_comparison_graph(league_data, selected_metric):
    """Create a league-level comparison graph across years"""
    metrics = {
        'K%': 'k_percent',
        'BB%': 'bb_percent',
        'Whiff%': 'whiff_percent',
        'Barrel%': 'barrel_percent',
        'Hard Hit%': 'hard_hit_percent',
        'xwOBA': 'xwoba'
    }

    metric_key = metrics[selected_metric]
    combined_league_data = pd.concat(league_data.values(), keys=league_data.keys(), names=['Group'])
    combined_league_data = combined_league_data.reset_index()

    fig = px.line(
        combined_league_data,
        x='year',
        y=metric_key,
        color='Group',
        title=f"League Averages for {selected_metric} Over Years",
        markers=True
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=selected_metric,
        plot_bgcolor='white'
    )
    return fig

def main():
    st.title("Pitcher Arm Angle Analysis")
    
    data, league_data = load_and_validate_data()
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

    # Filters for handedness and role
    st.subheader("Filter by Handedness and Role")
    handedness = st.selectbox("Select Handedness", ['RH', 'LH'])
    role = st.selectbox("Select Role", ['Starter', 'Reliever'])
    
    # Generate player-level graph
    fig = create_visualization(
        data[data['Role'] == role],
        league_data,
        selected_metric,
        bucket_size=bucket_size,
        plot_type=plot_type,
        handedness=handedness,
        role=role
    )
    if fig:
        st.plotly_chart(fig)
    
    # Generate league comparison graph
    st.subheader("League-Level Comparison")
    league_fig = create_league_comparison_graph(league_data, selected_metric)
    st.plotly_chart(league_fig)

