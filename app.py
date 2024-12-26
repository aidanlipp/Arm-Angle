def create_visualization(data, metric_name, metric_col, plot_type, bucket_size=None):
    """Create visualization with interactive selection"""
    if plot_type == "Scatter":
        fig = px.scatter(
            data,
            x='ball_angle',
            y=metric_col,
            color='year',
            title=f"{metric_name} vs Arm Angle",
            custom_data=['pitcher_name', 'year', 'k_percent', 'bb_percent', 
                        'whiff_percent', 'barrel_percent', 'hard_hit_percent', 'xwoba']
        )
        
        # Add click event template
        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Year: %{customdata[1]}<br>"
                "Arm Angle: %{x:.1f}°<br>"
                f"{metric_name}: %{y:.1f}<br>"
                "<i>Click for full stats</i><extra></extra>"
            )
        )
        
    else:  # Bar Chart
        data_with_buckets = create_angle_buckets(data.copy(), bucket_size)
        avg_data = data_with_buckets.groupby('angle_bucket').agg({
            metric_col: ['mean', 'count'],
            'pitcher_name': list
        }).reset_index()
        
        fig = px.bar(
            avg_data,
            x='angle_bucket',
            y=('metric_col', 'mean'),
            title=f"Average {metric_name} by Arm Angle ({bucket_size}° buckets)",
            text=avg_data[('metric_col', 'count')].apply(lambda x: f"n={x}"),
            custom_data=[('pitcher_name', 'list')]
        )

    # Add league average line
    league_avg = data[metric_col].mean()
    fig.add_hline(
        y=league_avg,
        line_dash="dash",
        line_color="red",
        annotation_text=f"League Avg: {league_avg:.1f}"
    )
    
    return fig

def main():
    # ... [previous code remains the same until visualization] ...

    # Create visualization
    fig = create_visualization(data, selected_metric, metrics[selected_metric]['col'], 
                             plot_type, bucket_size if plot_type != "Scatter" else None)
    
    # Add click event handling
    selected_point = plotly_events(fig, click_event=True)
    
    if selected_point:
        st.subheader("Selected Pitcher(s) Stats")
        if plot_type == "Scatter":
            # For scatter plot, show individual pitcher
            pitcher_name = selected_point[0]['customdata'][0]
            pitcher_data = data[data['pitcher_name'] == pitcher_name].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("K%", f"{pitcher_data['k_percent']:.1f}%")
                st.metric("BB%", f"{pitcher_data['bb_percent']:.1f}%")
            with col2:
                st.metric("Whiff%", f"{pitcher_data['whiff_percent']:.1f}%")
                st.metric("Barrel%", f"{pitcher_data['barrel_percent']:.1f}%")
            with col3:
                st.metric("Hard Hit%", f"{pitcher_data['hard_hit_percent']:.1f}%")
                st.metric("xwOBA", f"{pitcher_data['xwoba']:.3f}")
                
        else:  # Bar Chart
            # For bar chart, show all pitchers in bucket
            angle_bucket = selected_point[0]['x']
            bucket_pitchers = data[data['angle_bucket'] == angle_bucket]
            
            st.dataframe(
                bucket_pitchers[['pitcher_name', 'year', 'ball_angle', 'k_percent', 
                               'bb_percent', 'whiff_percent', 'barrel_percent', 
                               'hard_hit_percent', 'xwoba']]
                .sort_values('ball_angle')
                .set_index('pitcher_name')
                .round(2)
            )

if __name__ == "__main__":
    main()
