import plotly.express as px

def count_clusters(df):
    """
    Generate a bar plot showing the number of customers per cluster.

    Args:
        df (DataFrame): Input DataFrame containing the 'cluster' column.

    Returns:
        str: HTML representation of the bar plot.

    """
    # Group the dataframe by the cluster column
    counts = df['cluster'].value_counts()

    # Create a bar plot using Plotly Express
    fig = px.bar(x=counts.index, y=counts.values, labels={'x': 'Cluster', 'y': 'Count'},
                 title='Number of Customers per Cluster', color=counts.index)

    fig.update_layout(width=800, height=500)

    # Convert the plot to HTML
    plot_html = fig.to_html(full_html=False)

    return plot_html
