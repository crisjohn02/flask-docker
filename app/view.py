from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
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
from scipy.stats import ttest_ind

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

# Function to preprocess data based on the crosstab_config.json file
def preprocess_with_config(data, column, config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    variable_info = next((x for x in config['rows'] + config['columns'] if x['variable'] == column), None)
    if variable_info:
        values_mapping = {item['value']: item['label'] for item in variable_info['values']}
        return data.replace(values_mapping)
    else:
        return data

def get_column_label(column_name, config_file_path):
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    variable_info = next((x for x in config['rows'] + config['columns'] if x['variable'] == column_name), None)
    if variable_info:
        return variable_info['name']
    else:
        return column_name

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

    # Apply preprocessing based on the crosstab_config.json file
    for column in df.columns:
        df[column] = preprocess_with_config(df[column], column, 'static/crosstab_config2.json')

    # Get unique column names
    columns = df.columns.tolist()

    # Get unique values for each column
    unique_values = {column: df[column].unique().tolist() for column in df.columns}

    # Filter out non-categorical string columns
    categorical_columns = [col for col in df.columns if is_categorical(df[col])]

    # Load the crosstab configuration file
    with open('static/crosstab_config.json', 'r') as config_file:
        crosstab_config = json.load(config_file)

    # Pass the column names to the template
    return render_template('index.html', columns=columns, categorical_columns=categorical_columns, unique_values=unique_values, crosstab_config=crosstab_config)

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

        # Apply preprocessing based on the crosstab_config.json file
        for column in df.columns:
            df[column] = preprocess_with_config(df[column], column, 'static/crosstab_config2.json')

        # Get data for the selected column
        column_data = df[selected_column]

        # Get column names
        columns = df.columns.tolist()

        # Get unique values for each column
        unique_values = {column: df[column].unique().tolist() for column in df.columns}

        # Filter out non-categorical string columns
        categorical_columns = [col for col in df.columns if is_categorical(df[col])]

        # Get the label for the selected column based on the config file
        column_label = get_column_label(selected_column, 'static/crosstab_config2.json')

        # Prepare data for the table
        if not column_data.empty:
            column_data = preprocess_with_config(column_data, selected_column, 'static/crosstab_config2.json')
            column_value_counts = column_data.value_counts()
            total_count = len(column_data)
            column_data_table = [(value, count, f"{(count / total_count) * 100:.2f}%") for value, count in column_value_counts.items()]
        else:
            column_data_table = []

        # Generate the visualization based on the selected type
        plt.figure(figsize=(14, 8))  # Adjust the figure size as needed

        if visualization_type == 'bar':
            sns.countplot(x=selected_column, data=df, order=df[selected_column].value_counts().index)
            for i, v in enumerate(df[selected_column].value_counts()):
                plt.text(i, v + 0.5, str(v), ha='center')
            plt.xlabel(column_label, fontsize=12)  # Use the label from the config file
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        elif visualization_type == 'line':
            df[selected_column].value_counts().sort_index().plot(kind='line', marker='o')
            plt.xlabel(column_label, fontsize=12)  # Use the label from the config file
            plt.ylabel('Count', fontsize=12)
        elif visualization_type == 'pie':
            value_counts = df[selected_column].value_counts()
            plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
            plt.xlabel(column_label, fontsize=12)  # Use the label from the config file
        elif visualization_type == 'histogram':
            df[selected_column].plot(kind='hist', bins=10)
            plt.xlabel(column_label, fontsize=12)  # Use the label from the config file
            plt.ylabel('Frequency', fontsize=12)

        plt.tight_layout()  # Adjust layout for better spacing


        # Save the visualization to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert the image to base64 and encode it
        visualization = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Pass the encoded image to the template
        visualization_uri = f"data:image/png;base64,{visualization}"

        # Load the crosstab configuration file
        with open('static/crosstab_config.json', 'r') as config_file:
            crosstab_config = json.load(config_file)

        # Pass the visualization URI, column names, table data, and other data to the template
        return render_template('index.html', visualization=visualization_uri, categorical_columns=categorical_columns, unique_values=unique_values, column_label=column_label, selected_column=selected_column, crosstab_config=crosstab_config, columns=columns, column_data=column_data_table,)
  


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

        # Load the crosstab configuration file
        with open('static/crosstab_config.json', 'r') as config_file:
            crosstab_config = json.load(config_file)

        # Pass the column names to the template
        return render_template('crosstabs.html', crosstab_config=crosstab_config, columns=categorical_columns)

@app.route('/compute_crosstab', methods=['POST'])
def compute_crosstab():
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

        # Apply preprocessing based on the crosstab_config.json file
        for column in df.columns:
            df[column] = preprocess_with_config(df[column], column, 'static/crosstab_config2.json')

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
                # Calculate the frequency counts of unique values in selected_row
                selected_row_counts = df[selected_row].value_counts()
                total_selected_row_count = selected_row_counts.sum()

                # Calculate the percentage of each value in selected_row
                selected_row_percentage = (selected_row_counts / total_selected_row_count * 100).astype(float)
                selected_row_percentage_sum = selected_row_percentage.sum()

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
                                        'breakdown_table_percentage': breakdown_table_percentage,
                                        'selected_row_counts': selected_row_counts,
                                        'total_selected_row_count': total_selected_row_count,
                                        'selected_row_percentage': selected_row_percentage}
        elif computation_method == 't_test':
        # Perform T-test computation
            result = {}
            for selected_row in column_for_rows:
                group1 = df_selected[df_selected[selected_row] == 'group1'][column_for_columns]
                group2 = df_selected[df_selected[selected_row] == 'group2'][column_for_columns]

                t_statistic, p_value = ttest_ind(group1, group2)

                result[selected_row] = {'T-statistic': t_statistic, 'p-value': p_value}

        # Load the crosstab configuration file
        with open('static/crosstab_config.json', 'r') as config_file:
            crosstab_config = json.load(config_file)

        # Pass the result and selected columns to the template
        return render_template('crosstabs.html', crosstab_config=crosstab_config, selected_row_percentage_sum=selected_row_percentage_sum, total_selected_row_count=total_selected_row_count, total_column_value=total_column_value, result=result, columns=categorical_columns, column_for_columns=column_for_columns, column_for_rows_list=column_for_rows, computation_method=computation_method)
    


@app.route('/Ttest')
def Ttest():
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

        # Load the crosstab configuration file
        with open('static/crosstab_config.json', 'r') as config_file:
            crosstab_config = json.load(config_file)

        # Pass the column names to the template
        return render_template('Ttest.html', crosstab_config=crosstab_config, columns=categorical_columns)

@app.route('/compute_Ttest', methods=['POST'])
def compute_Ttest():
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

        # Apply preprocessing based on the crosstab_config.json file
        for column in df.columns:
            df[column] = preprocess_with_config(df[column], column, 'static/crosstab_config2.json')

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
        if computation_method == 't_test':
            # Perform T-test computation
            result = {}
            for selected_row in column_for_rows:
                # Calculate the frequency counts of unique values in selected_row
                selected_row_counts = df[selected_row].value_counts()
                total_selected_row_count = selected_row_counts.sum()

                # Calculate the percentage of each value in selected_row
                selected_row_percentage = (selected_row_counts / total_selected_row_count * 100).astype(float)
                selected_row_percentage_sum = selected_row_percentage.sum()

                contingency_table = pd.crosstab(df_selected[selected_row], df_selected[column_for_columns])
                # Construct contingency table for t-test
                group1 = df_selected[selected_row]
                group2 = df_selected[column_for_columns]

                # Perform t-test
                t_statistic, p_value = ttest_ind(group1, group2, equal_var=True)
                
                # Separate frequency and percentage in breakdown table
                breakdown_table = contingency_table.copy()
                breakdown_table_percentage = (contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100).astype(float)
                breakdown_percentage = (contingency_table.div(total_column_value, axis=1) * 100).astype(float)

                head_cell_count = 7  # Set a default value for head_cell_count
 
                # Check if selected_row is iterable (e.g., a list)
                if hasattr(selected_row, '__iter__'):
                    # Get the length of selected_row
                    selected_row_length = len(selected_row)
                    head_cell_count += selected_row_length
                
                # Add sum of rows and columns to the breakdown tables
                breakdown_table['Total'] = breakdown_table.sum(axis=1)
                breakdown_table_percentage['Total'] = (breakdown_percentage.sum(axis=1))
                breakdown_table.loc['Total'] = breakdown_table.sum()
                breakdown_table_percentage.loc['Total'] = breakdown_percentage.sum()
                
                result[selected_row] = {'t_statistic': t_statistic, 'p_value': p_value, 
                                        'breakdown_table_frequency': breakdown_table, 
                                        'breakdown_table_percentage': breakdown_table_percentage,
                                        'selected_row_counts': selected_row_counts,
                                        'total_selected_row_count': total_selected_row_count,
                                        'selected_row_percentage': selected_row_percentage}

        # Load the crosstab configuration file
        with open('static/crosstab_config.json', 'r') as config_file:
            crosstab_config = json.load(config_file)

        # Pass the result and selected columns to the template
        return render_template('Ttest.html', crosstab_config=crosstab_config, head_cell_count=head_cell_count, selected_row_percentage_sum=selected_row_percentage_sum, total_selected_row_count=total_selected_row_count, total_column_value=total_column_value, result=result, columns=columns, column_for_columns=column_for_columns, column_for_rows_list=column_for_rows, computation_method=computation_method)




if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
