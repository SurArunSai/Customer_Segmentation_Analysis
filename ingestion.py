import pandas as pd
from pymongo import MongoClient
from tabulate import tabulate
from IPython.display import display
import csv

class MongoDBLoader:
    def __init__(self, database_name, collection_name):
        """
        Initialize the MongoDBLoader class.

        Parameters:
        - database_name (str): The name of the MongoDB database.
        - collection_name (str): The name of the MongoDB collection.
        """
        self.database_name = database_name
        self.collection_name = collection_name

    def get_mongo_collection(self):
        """
        Get a MongoDB collection object.

        Returns:
        - collection (pymongo.collection.Collection): The MongoDB collection object.
        """
        client = MongoClient('mongodb://localhost:27017/')
        db = client[self.database_name]
        collection = db[self.collection_name]
        return collection

    def load_csv_to_mongodb(self, file_path):
        """
        Load data from a CSV file into MongoDB collection.

        Parameters:
        - file_path (str): The path to the CSV file.
        """
        collection = self.get_mongo_collection()

        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path, delimiter=';')

        # Convert the DataFrame records to a list of dictionaries
        data = df.to_dict('records')

        # Insert the data into the MongoDB collection
        collection.insert_many(data)


# Specify the MongoDB database name and collection name
database_name = 'BankMarketing'
collection_name = 'Telemarketing'

# Create an instance of MongoDBLoader
mongo_loader = MongoDBLoader(database_name, collection_name)

# Set the file path of your CSV file
file_path = '/home/arun/Master Of Data Science/Sem 3/Data Mining-CSC6004/Final/GUI/DataSets/bank-additional-full.csv'

# Load the CSV data into MongoDB
mongo_loader.load_csv_to_mongodb(file_path)