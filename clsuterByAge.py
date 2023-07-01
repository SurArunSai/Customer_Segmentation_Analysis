import pandas as pd
import plotly.graph_objects as go

def stacked_bar_chart(df):
    """
    Generate a stacked bar chart showing the distribution of clusters across different age groups.

    Args:
        df (DataFrame): Input DataFrame containing the 'age' and 'cluster' columns.

    Returns:
        str: HTML representation of the stacked bar chart.

    """
    # Group the dataframe by age and cluster
    df_grouped = df.groupby(['age', 'cluster']).size().unstack()

    # Create a stacked bar chart
    fig = go.Figure(data=[
        go.Bar(x=df_grouped.index, y=df_grouped[cluster], name=cluster) for cluster in df_grouped.columns
    ])

    # Update the layout with labels and title
    fig.update_layout(
        barmode='stack',
        xaxis_title='Age',
        yaxis_title='Count',
        title='Stacked Bar Chart: Cluster per Age'
    )

    # Convert the plot to HTML
    plot_html = fig.to_html(full_html=False)

    return plot_html
