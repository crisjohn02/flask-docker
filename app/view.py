from flask import Flask, render_template, request, jsonify, send_file
from flask_mysqldb import MySQL
from flask import make_response
import csv
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import io
import base64
from collections import defaultdict
import scipy.stats
app = Flask(__name__)

# Configure MySQL
app.config['MYSQL_HOST'] = 'host.docker.internal'
app.config['MYSQL_USER'] = 'local'
app.config['MYSQL_PASSWORD'] = 'secret'
app.config['MYSQL_DB'] = 'fluent'

# Initialize MySQL
mysql = MySQL(app)

# Function to preprocess data and exclude "null" values
def preprocess_data(data):
    return [val if val != "null" else None for val in data]

# Function to convert string values to numerical types (int or float)
def convert_to_numeric(data):
    try:
        return pd.to_numeric(data)
    except ValueError:
        return data

# Function to check if a column contains non-categorical string values
def is_categorical(column):
    try:
        pd.to_numeric(column)
        return True
    except ValueError:
        unique_values = column.unique()
        return len(unique_values) <= 10 and all(isinstance(val, str) for val in unique_values)

@app.route('/index')
def index():
    # Fetch data from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT `url_data` FROM `records`")
    data = cur.fetchall()
    cur.close()

    # Convert JSON data to DataFrame
    data_dicts = [json.loads(row[0]) for row in data]
    df = pd.DataFrame(data_dicts)

    # Preprocess data to combine "None" and "null" values
    for column in df.columns:
        df[column] = preprocess_data(df[column])

    # Convert string values to numerical types (int or float)
    df = df.apply(convert_to_numeric)

    # Get unique column names
    columns = df.columns.tolist()

    # Get unique values for each column
    unique_values = {column: df[column].unique().tolist() for column in df.columns}

    # Pass the column names to the template
    return render_template('index.html', columns=columns, unique_values=unique_values)


@app.route('/')
def home():
   
    # Pass the column names to the template
    return render_template('dashboard.html')


@app.route('/visualize_data', methods=['POST'])
def visualize_data():
    # Get the selected column name and visualization type from the form
    selected_column = request.form['column']
    visualization_type = request.form['visualization']

    # Fetch data from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT `url_data` FROM `records`")
    data = cur.fetchall()
    cur.close()

    # Convert JSON data to DataFrame
    data_dicts = [json.loads(row[0]) for row in data]
    df = pd.DataFrame(data_dicts)

    # Preprocess data to combine "None" and "null" values
    for column in df.columns:
        df[column] = preprocess_data(df[column])

    # Convert string values to numerical types (int or float)
    df = df.apply(convert_to_numeric)

    # Get data for the selected column
    column_data = df[selected_column]

    # Prepare data for the table
    if not column_data.empty:
        column_value_counts = column_data.value_counts()
        total_count = len(column_data)
        column_data_table = [(value, count, f"{(count / total_count) * 100:.2f}%") for value, count in column_value_counts.items()]
    else:
        column_data_table = []

    # Generate the visualization based on the selected type
    if visualization_type == 'bar':
        plt.figure(figsize=(10, 6))
        sns.countplot(x=selected_column, data=df, order=df[selected_column].value_counts().index)
        for i, v in enumerate(df[selected_column].value_counts()):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.xlabel(selected_column)
        plt.ylabel('Count')
    elif visualization_type == 'line':
        plt.figure(figsize=(10, 6))
        df[selected_column].value_counts().sort_index().plot(kind='line', marker='o')
        plt.xlabel(selected_column)
        plt.ylabel('Count')
    elif visualization_type == 'pie':
        plt.figure(figsize=(10, 6))
        value_counts = df[selected_column].value_counts()
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
        plt.xlabel(selected_column)
    elif visualization_type == 'histogram':
        plt.figure(figsize=(10, 6))
        df[selected_column].plot(kind='hist', bins=10)
        plt.xlabel(selected_column)
        plt.ylabel('Frequency')

    # Save the visualization to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convert the image to base64 and encode it
    visualization = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Pass the encoded image to the template
    visualization_uri = f"data:image/png;base64,{visualization}"

    # Get column names
    columns = df.columns.tolist()

    # Pass the visualization URI, column names, table data, and other data to the template
    return render_template('index.html', visualization=visualization_uri, selected_column=selected_column, columns=columns, column_data=column_data_table)


@app.route('/crosstabs')
def crosstabs():
    # Fetch data from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT `url_data` FROM `records`")
    data = cur.fetchall()
    cur.close()

    # Convert JSON data to DataFrame
    data_dicts = [json.loads(row[0]) for row in data]
    df = pd.DataFrame(data_dicts)

    # Preprocess data to combine "None" and "null" values
    for column in df.columns:
        df[column] = preprocess_data(df[column])

    # Convert string values to numerical types (int or float)
    df = df.apply(convert_to_numeric)

    # Get unique column names
    columns = df.columns.tolist()

    # Filter out non-categorical string columns
    categorical_columns = [col for col in df.columns if is_categorical(df[col])]

    # Pass the column names to the template
    return render_template('crosstabs.html', columns=categorical_columns)

@app.route('/compute_crosstab', methods=['POST'])
def compute_crosstab():
    try:
        # Get selected column names and computation method from the form
        column_for_columns = request.form['column_for_columns']
        column_for_rows = request.form.getlist('column_for_rows')  # Get list of selected rows
        computation_method = request.form['computation_method']

        # Fetch data from the database
        cur = mysql.connection.cursor()
        cur.execute("SELECT `url_data` FROM `records`")
        data = cur.fetchall()
        cur.close()

        # Convert JSON data to DataFrame
        data_dicts = [json.loads(row[0]) for row in data]
        df = pd.DataFrame(data_dicts)

        # Preprocess data to combine "None" and "null" values
        for column in df.columns:
            df[column] = preprocess_data(df[column])

        # Convert string values to numerical types (int or float)
        df = df.apply(convert_to_numeric)

        # Filter DataFrame based on selected columns
        df_selected = df[[column_for_columns] + column_for_rows]

        # Get unique column names
        columns = df.columns.tolist()

        # Filter out non-categorical string columns
        categorical_columns = [col for col in df.columns if is_categorical(df[col])]

        # Calculate the total value of the selected column
        total_column_value = df[column_for_columns].count()

        # Perform computation based on the selected method
        result = {}
        if computation_method == 'chi_square':
            # Perform Chi Square computation
            result = {}
            for selected_row in column_for_rows:
                contingency_table = pd.crosstab(df_selected[selected_row], df_selected[column_for_columns])
                chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency_table)
                
                # Separate frequency and percentage in breakdown table
                breakdown_table = contingency_table.copy()
                breakdown_table_percentage = (contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100).astype(float)
                breakdown_percentage = (contingency_table.div(total_column_value, axis=1) * 100).astype(float)
                
                # Add sum of rows and columns to the breakdown tables
                breakdown_table['Total'] = breakdown_table.sum(axis=1)
                breakdown_table_percentage['Total'] = (breakdown_percentage.sum(axis=1))
                breakdown_table.loc['Total'] = breakdown_table.sum()
                breakdown_table_percentage.loc['Total'] = breakdown_percentage.sum()
                
                result[selected_row] = {'Chi Square': chi2, 'p-value': p, 'Degrees of Freedom': dof, 
                                        'breakdown_table_frequency': breakdown_table, 
                                        'breakdown_table_percentage': breakdown_table_percentage}


        # Pass the result and selected columns to the template
        return render_template('crosstabs.html', total_column_value=total_column_value, result=result, columns=categorical_columns, column_for_columns=column_for_columns, column_for_rows_list=column_for_rows, computation_method=computation_method)
    except ValueError as e:
        error_message = 'An error occurred: ' + str(e)
        return render_template('crosstabs.html', error_message=error_message, columns=categorical_columns)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
