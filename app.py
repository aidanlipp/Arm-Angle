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
    
    # Add year selection
    available_years = sorted(data['year'].unique())
    selected_years = st.multiselect(
        "Select Years",
        options=available_years,
        default=available_years
    )
    
    # Filter data by selected years
    data = data[data['year'].isin(selected_years)]
    
    if plot_type == "Scatter":
        fig = px.scatter(
            data,
            x='ball_angle',
            y=metrics[selected_metric],
            color='year' if len(selected_years) > 1 else None,
            title=f"{selected_metric} vs Arm Angle",
            hover_data=['pitcher_name', 'year']
        )
        
        # Set color to blue if not coloring by year
        if len(selected_years) <= 1:
            fig.update_traces(marker=dict(color='blue'))
        
    else:  # Bar Chart
        data_with_buckets = create_angle_buckets(data.copy(), bucket_size)
        bucket_stats = data_with_buckets.groupby('angle_bucket').agg({
            metrics[selected_metric]: ['mean', 'count']
        }).reset_index()
        
        fig = px.bar(
            bucket_stats,
            x='angle_bucket',
            y=(metrics[selected_metric], 'mean'),
            title=f"Average {selected_metric} by Arm Angle ({bucket_size}° buckets)"
        )
        
        # Add counts as text on bars
        fig.update_traces(
            text=bucket_stats[(metrics[selected_metric], 'count')].apply(lambda x: f"n={x}"),
            textposition='auto'
        )
        
        # Add league average line
        league_avg = data[metrics[selected_metric]].mean()
        fig.add_hline(
            y=league_avg,
            line_dash="dash",
            line_color="red",
            annotation_text=f"League Avg: {league_avg:.1f}"
        )
    
    # Common layout updates
    fig.update_layout(
        xaxis_title="Arm Angle (degrees)",
        yaxis_title=selected_metric
    )
    
    selected_point = plotly_events(fig, click_event=True)
    
    if selected_point:
        st.subheader("Selected Pitcher(s)")
        display_columns = ['pitcher_name', 'year', 'ball_angle'] + list(metrics.values())
        if plot_type == "Bar Chart":
            bucket = selected_point[0]['x']
            df_display = data_with_buckets[data_with_buckets['angle_bucket'] == bucket]
        else:
            clicked_x = selected_point[0]['x']
            clicked_y = selected_point[0]['y']
            df_display = data[
                (abs(data['ball_angle'] - clicked_x) < 0.5) &
                (abs(data[metrics[selected_metric]] - clicked_y) < 0.5)
            ]
        
        if not df_display.empty:
            st.dataframe(
                df_display[display_columns]
                .sort_values('ball_angle')
                .round(2)
            )
