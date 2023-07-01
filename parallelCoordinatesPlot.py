import plotly.express as px

def parallel_coordinates_plot(df):
    """
    Create a parallel coordinates plot using Plotly Express.

    Args:
        df (DataFrame): Input DataFrame containing the data for the plot.

    Returns:
        str: HTML representation of the parallel coordinates plot.

    """
    # Create the parallel coordinates plot
    parallel_fig = px.parallel_coordinates(df, color='cluster')

    # Convert the plot to HTML
    plot_parallel_plotly = parallel_fig.to_html(full_html=False)

    return plot_parallel_plotly
