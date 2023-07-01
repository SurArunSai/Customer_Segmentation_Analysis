# Packages / libraries
from flask import Flask, render_template, request, Response, session
from flask_menu import Menu, register_menu
from flask_wtf.csrf import CSRFProtect
from flask_wtf import csrf
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from wtforms import ValidationError
import os #provides functions for interacting with the operating system
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, confusion_matrix, accuracy_score, classification_report, log_loss
from math import sqrt
from sklearn.metrics import silhouette_samples, silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.cluster import KMeans, k_means
from barchart import count_clusters
from clsuterByAge import stacked_bar_chart
from clusterPerJob import stacked_bar_chart_job
from clusterByMarital import stacked_bar_chart_marital
from clusterByEducation import stacked_bar_chart_education
from clusterByDefault import stacked_bar_chart_default
from clusterByHousing import stacked_bar_chart_housing
from clusterByLoanStatus import stacked_bar_chart_y
from clusterTree import create_cluster_tree
from parallelCoordinatesPlot import parallel_coordinates_plot
from KMeansClusteringAlgorithm import KMeansClusteringAlgorithm
from pymongo import MongoClient
import csv
# To install sklearn type "pip install numpy scipy scikit-learn" to the anaconda terminal

# To change scientific numbers to float
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Increases the size of sns plots
sns.set(rc={'figure.figsize':(8,6)})

app = Flask(__name__)
secret_key = os.urandom(16).hex()
app.config['SECRET_KEY'] = secret_key
csrf = CSRFProtect(app)

class MyForm(FlaskForm):
    input_number = 0
    input_number = StringField('Input Number', validators=[DataRequired()])
    input_iterations = 0
    input_iterations = StringField('Input Iterations', validators=[DataRequired()])
    def validate_input_number(self, field):
        try:
            int(field.data)
        except ValueError:
            raise ValidationError('Input must be a numeric value.')
    submit = SubmitField('Submit')


@app.route('/', methods=['GET', 'POST'])
def index():
    plot_parallel = None
    plot_count_clusters = None
    plot_count_clusters_by_age = None
    plot_count_clusters_by_job = None
    plot_count_clusters_by_marital = None
    plot_count_clusters_by_education = None
    plot_count_clusters_by_default = None
    plot_count_clusters_by_housing = None
    plot_count_clusters_by_y = None
    plot_cluster_tree = None
    plot_cluster_tree_1 = None
    plot_cluster_tree_2 = None
    total_rows = "Not Calculated"
    table = None
    form = MyForm()
    if form.validate_on_submit():
        input_number = form.input_number.data
        input_number = int(input_number)
        input_iterations = form.input_iterations.data
        input_iterations = int(input_iterations)
        analyzer = KMeansClusteringAlgorithm(data_path)
        analyzer.import_raw_data()
        analyzer.preprocess_data()
        labels, df = analyzer.cluster_data(max_iterations=input_iterations, centroid_count=input_number)
        raw_df = df.drop(['poutcome', 'emp.var.rate', 'cons.price.idx',
        'cons.conf.idx', 'euribor3m', 'nr.employed', 'duration','campaign', 'previous', 'pdays'], axis=1)
        # This data is converted from categorical data to numreric for vizuvalization
        df = df.drop(['_id', 'duration', 'campaign',
       'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed'], axis=1)
        df['cluster'] = labels        
        # Convert DataFrame to HTML table
        table_data = raw_df.head(10)
        table = table_data.to_html(classes='table table-bordered')
        
        # Total Number of Clsuters
        total_rows = df.shape[0]    

        # Bar chart for clusters
        plot_count_clusters = count_clusters(raw_df)

        # Stacked Bar Chart for Age by Clsuters
        plot_count_clusters_by_age = stacked_bar_chart(raw_df)

        # Stacked Bar Chart for Job by Clusters
        plot_count_clusters_by_job = stacked_bar_chart_job(raw_df)

        # Stacked Bar Chart for Marital by Clusters
        plot_count_clusters_by_marital = stacked_bar_chart_marital(raw_df)

        # Stacked Bar Chart for Education by Clusters
        plot_count_clusters_by_education = stacked_bar_chart_education(raw_df)

        # Stacked Bar Chart for Default by Clusters
        plot_count_clusters_by_default = stacked_bar_chart_default(raw_df)

        # Stacked Bar Chart for Housing by Clusters
        plot_count_clusters_by_housing = stacked_bar_chart_housing(raw_df)

        # Stacked Bar Chart for LoanStatus by Clusters
        plot_count_clusters_by_y = stacked_bar_chart_y(raw_df)

        # Cluster Trees
        plot_cluster_tree, plot_cluster_tree_1, plot_cluster_tree_2 = create_cluster_tree(raw_df)

        #Parallel Coordinates
        plot_parallel = parallel_coordinates_plot(df)

    return render_template('index.html', 
    plot_parallel_html=plot_parallel, 
    plot_count_clusters_html=plot_count_clusters, 
    plot_count_cluster_by_age_html=plot_count_clusters_by_age, 
    plot_count_clusters_by_job_html = plot_count_clusters_by_job,
    plot_count_clusters_by_marital_html = plot_count_clusters_by_marital,
    plot_count_clusters_by_education_html = plot_count_clusters_by_education,
    plot_count_clusters_by_default_html = plot_count_clusters_by_default,
    plot_count_clusters_by_housing_html = plot_count_clusters_by_housing,
    plot_count_clusters_by_y_html = plot_count_clusters_by_y,
    plot_cluster_tree_html = plot_cluster_tree,
    plot_cluster_tree_1_html = plot_cluster_tree_1,
    plot_cluster_tree_2_html = plot_cluster_tree_2,
    total_rows=total_rows,
    table=table,
    form=form
    )

@app.route('/main')
def main_page():
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True)