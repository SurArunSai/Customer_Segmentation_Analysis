import pandas as pd
import plotly.graph_objects as go

def stacked_bar_chart_y(df):
    """
    Generate a stacked bar chart showing the distribution of clusters across different y categories.

    Args:
        df (DataFrame): Input DataFrame containing the 'y' and 'cluster' columns.

    Returns:
        str: HTML representation of the stacked bar chart.

    """
    # Group the dataframe by y and cluster
    df_grouped = df.groupby(['y', 'cluster']).size().unstack()

    # Create a stacked bar chart
    fig = go.Figure(data=[
        go.Bar(x=df_grouped.index, y=df_grouped[cluster], name=cluster) for cluster in df_grouped.columns
    ])

    # Update the layout with labels and title
    fig.update_layout(
        barmode='stack',
        xaxis_title='y',
        yaxis_title='Count',
        title='Stacked Bar Chart: Cluster per y'
    )

    # Convert the plot to HTML
    plot_html = fig.to_html(full_html=False)

    return plot_html
