import pandas as pd
import plotly.express as px

def create_cluster_tree(df):
    """
    Create cluster tree visualizations using sunburst plots.

    Args:
        df (DataFrame): Input DataFrame containing the relevant columns for visualization.

    Returns:
        tuple: HTML representations of the sunburst plots.

    """
    # Create the first sunburst plot
    fig = px.sunburst(df, path=['education', 'job', 'marital', 'y'], values='cluster')

    # Create the second sunburst plot
    fig_1 = px.sunburst(df, path=['marital', 'loan', 'housing', 'education'], values='cluster')

    # Create the third sunburst plot
    fig_2 = px.sunburst(df, path=['education', 'marital', 'default', 'age'], values='cluster')

    # Convert the plots to HTML
    plot_html = fig.to_html(full_html=False)
    plot_html_1 = fig_1.to_html(full_html=False)
    plot_html_2 = fig_2.to_html(full_html=False)

    return plot_html, plot_html_1, plot_html_2
