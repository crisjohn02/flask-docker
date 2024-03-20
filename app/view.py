from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session, make_response, Response
from flask_mysqldb import MySQL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats
from scipy.stats import f_oneway, pearsonr, spearmanr, ttest_ind, chi2_contingency
import csv
import os
import json
import io
import base64
import socket
import math
from itertools import product
from collections import defaultdict
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure MySQL
app.config['MYSQL_HOST'] = 'host.docker.internal'
app.config['MYSQL_USER'] = 'dan'
app.config['MYSQL_PASSWORD'] = 'dan'
app.config['MYSQL_DB'] = 'fluent'

# Initialize MySQL
mysql = MySQL(app)


# Get config
def get_config():
    with open('static/crosstab_config.json', 'r') as config_file:
        config = json.load(config_file)

    return config

# Get the config row/column element
def get_element(config, variable, value, el_type='rows'):
    # Create a dictionary to map variables to their values arrays
    variable_values_map = {el['variable']: el['values'] for el in config[el_type]}

    values = variable_values_map.get(variable)
    if values:
        for item in values:
            if item['value'] == value:
                return item
    return {'label': None}


# Function to get label based on variable and value
def get_label(variable, value, config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    for item in config['survey']:
        if item['variable'] == variable:
            for value_item in item.get('values', []):
                if value_item['value'] == value:
                    return value_item['label']
    return value  # Return original value if not found in survey config

def process_value(value):
    if value in (None, "null", "", "-"):
        return None
    elif isinstance(value, str) and value.isdigit():
        return int(value)
    else:
        return value

# Function to get the name and data type of the selected column based on the config file
def get_column_info(selected_columns, config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    variable_info = next((x for x in config['survey'] if x['variable'] == selected_columns), None)
    if variable_info:
        name = variable_info['name']
        data_type = variable_info.get('data_type', None)
        return name, data_type
    else:
        return None, None
    
# Function to get the name and data type of the selected column based on the config file
def get_row_info(selected_columns, config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    variable_info = next((x for x in config['rows'] if x['variable'] == selected_columns), None)
    if variable_info:
        name = variable_info['name']
        return name
    else:
        return None

# Function to get the name and data type of the selected column based on the config file
def get_column_info(selected_columns, config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    variable_info = next((x for x in config['columns'] if x['variable'] == selected_columns), None)
    if variable_info:
        name = variable_info['name']
        return name
    else:
        return None
    
def get_column_label(variable, value, config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    for item in config['columns']:
        if item['variable'] == variable:
            for value_item in item.get('values', []):
                if value_item['value'] == value:
                    return value_item['label']
    return value  

def get_row_label(variable, value, config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    for item in config['rows']:
        if item['variable'] == variable:
            for value_item in item.get('values', []):
                if value_item['value'] == value:
                    return value_item['label']
    return value  # Return original value if not found in survey config

@app.route('/alldata', methods=['GET'])
def alldata():
    # Fetch data from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT url_data FROM records WHERE survey_code='lQuDql' AND status='cp' AND test_id=0")
    rows = cur.fetchall()
    cur.close()

    # Process fetched data
    data_dict = {}
    max_length = 0  # Track the maximum length of lists

    for row in rows:
        url_data_str = row[0]
        # Parse the string into a dictionary
        url_data = json.loads(url_data_str)
        for key, value in url_data.items():
            # Check if the value is None or "null"
            value = None if value in (None, "null", "") else value  # Replace empty values with None
            # Convert numeric strings into integers
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(value)
            max_length = max(max_length, len(data_dict[key]))

    # Pad shorter lists with None to match the length of the longest list
    for key in data_dict:
        data_dict[key] += [None] * (max_length - len(data_dict[key]))

    # Convert dictionary to JSON and return with headers
    response = jsonify(data_dict)
    return response

@app.route('/datalist', methods=['GET'])
def datalist():
    # Fetch data from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT url_data FROM records WHERE survey_code='lQuDql' AND status='cp' AND test_id=0")
    rows = cur.fetchall()
    cur.close()

    # Process fetched data
    data_dict = {}
    for row in rows:
        url_data_str = row[0]
        # Parse the string into a dictionary
        url_data = json.loads(url_data_str)
        for key in url_data.keys():
            if key not in data_dict:
                data_dict[key] = []

    # Convert dictionary to the desired structure
    question_array = list(data_dict.keys())

    # Construct the response JSON
    response_data = {"question": question_array}

    # Convert dictionary to JSON and return with headers
    response = jsonify(response_data)
    return response

@app.route('/data/<selected_data>', methods=['GET'])
def data(selected_data):
    # Split the selected data by comma to get individual column names
    selected_columns = selected_data.split(',')

    # Fetch data from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT url_data FROM records WHERE survey_code='lQuDql' AND status='cp' AND test_id=0")
    rows = cur.fetchall()
    cur.close()

    # Process fetched data based on selected columns
    data_dict = {column: [] for column in selected_columns}
    max_length = 0 
    for row in rows:
        url_data = json.loads(row[0])
        for column in selected_columns:
            value = url_data.get(column)
            if value == "null":
                value = None
            elif value == "":
                value = None
            elif isinstance(value, str) and value.isdigit():
                value = int(value)
            data_dict[column].append(value)
            max_length = max(max_length, len(data_dict[column]))

    for column in data_dict:
        data_dict[column] += [None] * (max_length - len(data_dict[column]))

    # Convert dictionary to JSON and return
    return jsonify(data_dict)

@app.route('/visualize_data/<selected_data>/<visualization_type>', methods=['GET', 'POST'])
def visualize_data(selected_data, visualization_type):
    selected_column = selected_data.split(',')

    # Fetch data from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT url_data FROM records WHERE survey_code='lQuDql' AND status='cp' AND test_id=0")
    rows = cur.fetchall()
    cur.close()

    # Process fetched data
    data_dict = {}
    max_length = 0  # Track the maximum length of lists

    for row in rows:
        url_data_str = row[0]
        # Parse the string into a dictionary
        url_data = json.loads(url_data_str)
        for key, value in url_data.items():
            # Check if the value is None or "null"
            value = None if value in (None, "null", "") else value  # Replace empty values with None
            # Convert numeric strings into integers
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(value)
            max_length = max(max_length, len(data_dict[key]))

    # Pad shorter lists with None to match the length of the longest list
    for key in data_dict:
        data_dict[key] += [None] * (max_length - len(data_dict[key]))

    df = pd.DataFrame(data_dict)

    # Get the name and data type for the selected column based on the config file
    column_name, column_data_type = get_column_info(selected_column[0], 'static/survey_config.json')

    # Generate the visualization based on the selected type
    plt.figure(figsize=(8, 4))  # Adjust the figure size as needed

    if visualization_type == 'bar':
        sns.countplot(x=selected_column[0], data=df, order=df[selected_column[0]].value_counts().index)
        for i, v in enumerate(df[selected_column[0]].value_counts()):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.xlabel(column_name, fontsize=16)  # Use the name from the config file
        plt.ylabel('Count', fontsize=16)
        plt.xticks(rotation=0, ha='right')  # Rotate x-axis labels for better readability
    elif visualization_type == 'line':
        df[selected_column[0]].value_counts().sort_index().plot(kind='line', marker='o')
        plt.xlabel(column_name, fontsize=16)  # Use the name from the config file
        plt.ylabel('Count', fontsize=16)
        plt.xticks(rotation=0, ha='right')  # Rotate x-axis labels for better readability
    elif visualization_type == 'pie':
        value_counts = df[selected_column[0]].value_counts()
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
        plt.xlabel(column_name, fontsize=16)  # Use the name from the config file
    elif visualization_type == 'histogram':
        df[selected_column[0]].plot(kind='hist', bins=10)
        plt.xlabel(column_name, fontsize=16)  # Use the label from the config file
        plt.ylabel('Frequency', fontsize=16)
        plt.xticks(rotation=0, ha='right')  # Rotate x-axis labels for better readability

    plt.tight_layout()  # Adjust layout for better spacing

    # Save the visualization to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convert the image to base64 and encode it
    visualization = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Pass the encoded image to the template
    visualization_uri = f"data:image/png;base64,{visualization}"
        
    return visualization_uri

@app.route('/visualize_table/<selected_data>', methods=['GET', 'POST'])
def visualize_table(selected_data):
    selected_column = selected_data.split(',')

    # Fetch data from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT url_data FROM records WHERE survey_code='lQuDql' AND status='cp' AND test_id=0")
    rows = cur.fetchall()
    cur.close()

    # Process fetched data
    data_dict = {}
    max_length = 0  # Track the maximum length of lists

    for row in rows:
        url_data_str = row[0]
        # Parse the string into a dictionary
        url_data = json.loads(url_data_str)
        for key, value in url_data.items():
            # Check if the value is None or "null"
            value = None if value in (None, "null", "") else value  # Replace empty values with None
            # Convert numeric strings into integers
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(value)
            max_length = max(max_length, len(data_dict[key]))

    # Pad shorter lists with None to match the length of the longest list
    for key in data_dict:
        data_dict[key] += [None] * (max_length - len(data_dict[key]))

    df = pd.DataFrame(data_dict)

    # Get the name and data type for the selected column based on the config file
    column_name, _ = get_column_info(selected_column[0], 'static/survey_config.json')

    # Calculate column data (value, count, percentage) for the selected column
    if selected_column[0] in df.columns:
        column_data = df[selected_column[0]].value_counts().reset_index()
        column_data.columns = [column_name, 'Count']
        total_count = column_data['Count'].sum()
        column_data['Percentage'] = (column_data['Count'] / total_count) * 100
        column_data = column_data.values.tolist()
    else:
        column_data = None

    return jsonify(column_data)

    
    

#CROSSTAB AND CHI-SQUARE API
# Crosstab API for Vue.js
def process_crosstabs(records, config):
    data_dict = {}
    max_length = 0

    for record in records:
        url_data = json.loads(record[0])
        for key, value in url_data.items():
            if value in (None, "null", "", "-"):
                value = None
            elif isinstance(value, str) and value.isdigit():
                value = int(value)
            data_dict.setdefault(key, []).append(value)
            max_length = max(max_length, len(data_dict[key]))

    for key in data_dict:
        data_dict[key] += [None] * (max_length - len(data_dict[key]))

    df = pd.DataFrame(data_dict)

    results = {}
    for columns in config['columns']:
        columns_var = columns['variable']
        labels_columns = df[columns_var].apply(lambda x: get_element(config, columns_var, x, 'columns')['label'])
        for rows in config['rows']:
            rows_var = rows['variable']
            labels_rows = df[rows_var].apply(lambda x: get_element(config, rows_var, process_value(x), 'rows')['label'])

            crosstab_result = pd.crosstab(labels_columns, labels_rows, margins=True, margins_name='Total')
            crosstab_result_row_percent = pd.crosstab(labels_columns, labels_rows, normalize='index')
            crosstab_result_column_percent = pd.crosstab(labels_columns, labels_rows, normalize='columns')
            crosstab_result_total_percent = pd.crosstab(labels_columns, labels_rows, normalize='all', margins=True, margins_name='Total')

            # Calculate row totals
            row_totals = crosstab_result_row_percent.sum(axis=1)
            total_row = pd.DataFrame({'Total': [1] * len(row_totals)}, index=row_totals.index)
            crosstab_result_row_percent = pd.concat([crosstab_result_row_percent, total_row], axis=1)

            # Calculate column totals
            column_totals = crosstab_result_column_percent.sum()
            total_column = pd.DataFrame({'Total': [1] * len(column_totals)}, index=column_totals.index).transpose()
            crosstab_result_column_percent = pd.concat([crosstab_result_column_percent, total_column])

            chi2_stat, p_val, dof, expected = chi2_contingency(crosstab_result)

            results[f"{columns_var},{rows_var}"] = {
                'crosstab': crosstab_result.to_dict(),
                'row_percentage': crosstab_result_row_percent.to_dict(),
                'column_percentage': crosstab_result_column_percent.to_dict(),
                'total_percentage': crosstab_result_total_percent.to_dict(),
                'chi_square_statistic': float(chi2_stat),
                'degrees_of_freedom': int(dof),
                'p_value': float(p_val)
            }

    return results


@app.route('/crosstabs', methods=['GET', 'POST'])
def crosstabs():
    cur = mysql.connection.cursor()
    cur.execute("SELECT url_data FROM records WHERE survey_code='lQuDql' AND status='cp' AND test_id=0")
    records = cur.fetchall()
    cur.close()

    # get the config as dict
    config = get_config()

    results = process_crosstabs(records, config)

    # Convert dictionary to JSON string
    json_string = json.dumps(results)

    # Return JSON string with proper content type
    return Response(json_string, content_type='application/json')


# Export to CSV API
def process_crosstabs_csv(records, config):
    data_dict = {}
    max_length = 0  
    
    for record in records:
        url_data = json.loads(record[0])
        for key, value in url_data.items():
            if value in (None, "null", "", "-"):
                value = None
            elif isinstance(value, str) and value.isdigit():
                value = int(value)
            data_dict.setdefault(key, []).append(value)
            max_length = max(max_length, len(data_dict[key]))

    for key in data_dict:
        data_dict[key] += [None] * (max_length - len(data_dict[key]))

    df = pd.DataFrame(data_dict)

    results = {}
    columns_var_counts = {}
    rows_var_counts = {}
    rows_var_percentage = {}

    for columns in config['columns']:
        columns_var = columns['variable']
        columns_var_counts[columns_var] = df[columns_var].value_counts()
        labels_columns = df[columns_var].apply(lambda x: get_element(config, columns_var, x, 'columns')['label'])
        for rows in config['rows']:
            rows_var = rows['variable']
            labels_rows = df[rows_var].apply(lambda x: get_element(config, rows_var, process_value(x), 'rows')['label'])
            rows_var_counts[rows_var] = labels_rows.value_counts().sort_index()  # Sorting counts by index
            rows_var_percentage[rows_var] = ((rows_var_counts[rows_var]) / len(df[rows_var])) * 100

            crosstab_result = pd.crosstab(labels_rows, labels_columns, margins=True, margins_name='Total')
            crosstab_result_row_percent = pd.crosstab(labels_rows, labels_columns, normalize='index')
            crosstab_result_column_percent = pd.crosstab(labels_rows, labels_columns, normalize='columns')
            crosstab_result_total_percent = pd.crosstab(labels_rows, labels_columns, normalize='all', margins=True, margins_name='Total')

            # Calculate row totals
            row_totals = crosstab_result_row_percent.sum(axis=1)
            total_row = pd.DataFrame({'Total': [1] * len(row_totals)}, index=row_totals.index)
            crosstab_result_row_percent = pd.concat([crosstab_result_row_percent, total_row], axis=1)

            # Calculate column totals
            column_totals = crosstab_result_column_percent.sum()
            total_column = pd.DataFrame({'Total': [1] * len(column_totals)}, index=column_totals.index).transpose()
            crosstab_result_column_percent = pd.concat([crosstab_result_column_percent, total_column])

            chi2_stat, p_val, _, _ = chi2_contingency(crosstab_result)
            degrees_of_freedom = (crosstab_result.shape[0] - 2) * (crosstab_result.shape[1] - 2)

            if columns_var not in results:
                results[columns_var] = {}
            results[columns_var][rows_var] = {
                'crosstab': crosstab_result,
                'row_percentage': crosstab_result_row_percent,
                'column_percentage': crosstab_result_column_percent,
                'total_percentage': crosstab_result_total_percent,
                'chi_square_statistic': chi2_stat,
                'degrees_of_freedom': degrees_of_freedom,
                'p_value': p_val
            }

    return results, columns_var_counts, rows_var_counts, rows_var_percentage

@app.route('/export_csv/<csv_outputs>', methods=['GET'])
def export_crosstabs_csv(csv_outputs):
    cur = mysql.connection.cursor()
    cur.execute("SELECT url_data FROM records WHERE survey_code='lQuDql' AND status='cp' AND test_id=0")
    records = cur.fetchall()
    cur.close()

    # get the config as dict
    config = get_config()

    results, columns_var_counts, rows_var_counts, rows_var_percentage = process_crosstabs_csv(records, config)

    # Check if csv_output parameter is valid
    valid_outputs = ['frequency_crosstab', 'row_percentage', 'column_percentage', 'total_percentage', 'total_crosstab', 'chi_square_results', 'row_crosstab', 'column_crosstab']
    selected_outputs = csv_outputs.split(',')
    for output in selected_outputs:
        if output not in valid_outputs:
            return f"Invalid csv_output parameter: {output}", 400
        

    # Export selected outputs to CSV
    csv_data = []
    
    for columns in config['columns']:
        columns_var = columns['variable']
        # columns_name = get_column_info(columns_var, 'static/crosstab_config.json')
        columns_name = columns['name']
        csv_data.append([''] + [''] + ['All'] + [''] + [''] + [f'"{columns_name}"'])
        
        for rows in config['rows']:
            rows_var = rows['variable']
            # rows_name = get_row_info(rows_var, 'static/crosstab_config.json')
            rows_name = rows['name']
            
            for csv_output in selected_outputs:
                if csv_output == 'frequency_crosstab':
                    crosstab = results[columns_var][rows_var]['crosstab']
                    rows_labels = [''] + [''] * 3 + [f'"{label}"' for label in crosstab.columns.tolist()]
                    csv_data.append([f'"{rows_name}"'])
                    csv_data.append(rows_labels)
                    dep_values = rows_var_counts[rows_var]
                    dep_values_subset = dep_values[:(len(crosstab) - 1)]
                    dep_values_sum = sum(dep_values_subset)
                    for (index, row), dep in zip(crosstab.iterrows(), dep_values_subset):
                        if index != crosstab.index[-1]: 
                            csv_data.append([f'"{index}"'] + ['Frequency'] + [dep] + ['-'] + row.values.tolist())

                    last_index_value = f'"{crosstab.index[-1]}"'  
                    last_row_values = list(crosstab.iloc[-1])
                    csv_data.append([last_index_value] + ['Frequency'] + [dep_values_sum] + [dep_values_sum] + last_row_values)
                    csv_data.append([''])

                elif csv_output in ['row_percentage', 'column_percentage', 'total_percentage']:
                    percentage_type = csv_output.split('_')[0].capitalize()
                    percentage_data = results[columns_var][rows_var][csv_output]
                    rows_labels = [''] + [''] * 3 + [f'"{label}"' for label in percentage_data.columns.tolist()]
                    csv_data.append([f'"{rows_name}"'])
                    csv_data.append(rows_labels)
                    crosstab = results[columns_var][rows_var]['crosstab']
                    dep_values = rows_var_counts[rows_var]
                    dep_values_subset = dep_values[:(len(crosstab) - 1)]
                    dep_values_sum = sum(dep_values_subset)
                    dep_values_percentage = (dep_values_subset / dep_values_sum) * 100
                    dep_percent_sum = (dep_values_sum / dep_values_sum) * 100
                    
                    for (index, row), percent in zip(percentage_data.iterrows(), dep_values_percentage):
                        if csv_output == 'column_percentage':
                            csv_data.append([f'"{index}"'] + ['Column%'] + [f'{percent:.2f}%'] + ['-'] + [f"{value * 100:.2f}%" for value in row.values.tolist()])
                        elif csv_output == 'row_percentage':
                            csv_data.append([f'"{index}"'] + ['Row%'] + [f'{percent:.2f}%'] + ['-'] + [f"{value * 100:.2f}%" for value in row.values.tolist()])
                        elif csv_output == 'total_percentage':
                            csv_data.append([f'"{index}"'] + ['Total%'] + [f'{percent:.2f}%'] + ['-'] + [f"{value * 100:.2f}%" for value in row.values.tolist()])
                    
                    last_index_value = f'"{percentage_data.index[-1]}"'  
                    last_row_values = list(percentage_data.iloc[-1])
                    if csv_output == 'row_percentage':
                        csv_data.append(['Total'] + [f'{percentage_type}%'] + [f'{dep_percent_sum:.2f}%'] + [f'{dep_percent_sum:.2f}%'] + ['-' for value in last_row_values] )
                    else: 
                        csv_data.append([last_index_value] + [f'{percentage_type}%'] + [f'{dep_percent_sum:.2f}%'] + [f'{dep_percent_sum:.2f}%'] + [f"{value * 100:.2f}%" for value in last_row_values] )
                    csv_data.append([''])

                elif csv_output == 'chi_square_results':
                    csv_data.append([f'"{rows_name} vs {columns_name}"'])
                    p_value = results[columns_var][rows_var]['p_value']
                    chi_square_statistic = results[columns_var][rows_var]['chi_square_statistic']
                    degrees_of_freedom = results[columns_var][rows_var]['degrees_of_freedom']
                    significance = "<" if p_value < 0.05 else ">"
                    significance_text = "significant" if p_value < 0.05 else "non-significant"
                    total_values = (columns_var_counts[columns_var]).sum()
                    csv_data.append([f'"The association between {rows_name} and which best describes your {columns_name}? is {significance_text} X\u00B2 ({total_values}) = {chi_square_statistic:.2f}, df = {degrees_of_freedom}, p {significance} 0.5"'])
                    csv_data.append([''])
                
                elif csv_output in ['row_crosstab', 'column_crosstab', 'total_crosstab']:
                    if csv_output == 'total_crosstab':
                        percentage_type = results[columns_var][rows_var]['total_percentage']
                    elif csv_output == 'row_crosstab':
                        percentage_type = results[columns_var][rows_var]['row_percentage']
                    elif csv_output == 'column_crosstab':   
                        percentage_type = results[columns_var][rows_var]['column_percentage']

                    crosstab = results[columns_var][rows_var]['crosstab']
                    crosstab_iterator = crosstab.iterrows()
                    dep_values = rows_var_counts[rows_var]
                    p_value = results[columns_var][rows_var]['p_value']
                    chi_square_statistic = results[columns_var][rows_var]['chi_square_statistic']
                    degrees_of_freedom = results[columns_var][rows_var]['degrees_of_freedom']
                    significance = "<" if p_value < 0.05 else ">"
                    significance_text = "significant" if p_value < 0.05 else "non-significant"
                    total_values = (columns_var_counts[columns_var]).sum()
                    rows_labels = [''] + [''] + [''] + [''] + [f'"{label}"' for label in crosstab.columns.tolist()]
                    csv_data.append([f'"{rows_name}"'])
                    csv_data.append(rows_labels)
                    
                    dep_values = rows_var_counts[rows_var]
                    dep_values_subset = dep_values[:(len(crosstab) - 1)]
                    dep_values_sum = sum(dep_values_subset)
                    dep_values_percentage = (dep_values_subset / dep_values_sum) * 100
                    dep_percent_sum = (dep_values_sum / dep_values_sum) * 100
                    last_index_value = f'"{crosstab.index[-1]}"'  
                    last_row_values = list(crosstab.iloc[-1])
                    _last_row_values = list(percentage_type.iloc[-1])

                    if csv_output == 'total_crosstab':
                        total_percentage_iterator = percentage_type.iterrows()
                        for (index, (crosstab_index, crosstab_row)), dep, (total_percentage_index, total_percentage_row), percent in zip(enumerate(crosstab_iterator), dep_values_subset, total_percentage_iterator, dep_values_percentage):
                            csv_data.append([f'"{crosstab_index}"'] + ['Frequency'] + [dep] + ['-'] + crosstab_row.values.tolist())
                            csv_data.append([''] + ['Total%'] + [f'{percent:.2f}%'] + ['-'] + [f'"{(value*100):.2f}%"' for value in total_percentage_row.tolist()])
                        csv_data.append([last_index_value] + ['Frequency'] + [dep_values_sum] + [dep_values_sum] + last_row_values)
                        csv_data.append([''] + ['Total%'] + [f'{dep_percent_sum:.2f}%'] + [f'{dep_percent_sum:.2f}%'] + [f"{value * 100:.2f}%" for value in _last_row_values] )
                        
                    elif csv_output == 'row_crosstab':
                        row_percentage_iterator = percentage_type.iterrows()
                        for (index, (crosstab_index, crosstab_row)), dep, (row_percentage_index, row_percentage_row), percent in zip(enumerate(crosstab_iterator), dep_values_subset, row_percentage_iterator, dep_values_percentage):
                            csv_data.append([f'"{crosstab_index}"'] + ['Frequency'] + [dep] + ['-'] + crosstab_row.values.tolist())
                            csv_data.append([''] + ['Row%'] + [f'{percent:.2f}%'] + ['-'] + [f'"{(value*100):.2f}%"' for value in row_percentage_row.tolist()])
                        csv_data.append([last_index_value] + ['Frequency'] + [dep_values_sum] + [dep_values_sum] + last_row_values)
                        csv_data.append([''] + ['Row%'] + [f'{dep_percent_sum:.2f}%'] + [f'{dep_percent_sum:.2f}%'] + ['-' for value in _last_row_values] )
                        
                    elif csv_output == 'column_crosstab':
                        column_percentage_iterator = percentage_type.iterrows()
                        for (index, (crosstab_index, crosstab_row)), dep, (column_percentage_index, column_percentage_row), percent in zip(enumerate(crosstab_iterator), dep_values_subset, column_percentage_iterator, dep_values_percentage):
                            csv_data.append([f'"{crosstab_index}"'] + ['Frequency'] + [dep] + ['-'] + crosstab_row.values.tolist())
                            csv_data.append([''] + ['Column%'] + [f'{percent:.2f}%'] + ['-'] + [f'"{(value*100):.2f}%"' for value in column_percentage_row.tolist()])
                        csv_data.append([last_index_value] + ['Frequency'] + [dep_values_sum] + [dep_values_sum] + last_row_values)
                        csv_data.append([''] + ['Column%'] + [f'{dep_percent_sum:.2f}%'] + [f'{dep_percent_sum:.2f}%'] + [f"{value * 100:.2f}%" for value in _last_row_values] )

                    csv_data.append([f'"The association between {rows_name} and which best describes your {columns_name}? is {significance_text} X\u00B2 ({total_values}) = {chi_square_statistic:.2f}, df = {degrees_of_freedom}, p {significance} 0.5"'])
                    csv_data.append([''])
            csv_data.append([''])
        csv_data.append([""])

    # Generate CSV string
    csv_string = "\ufeff" + "\n".join([",".join(map(str, row)) for row in csv_data])

    # Set filename based on selected csv_outputs
    # filename = f"{'_'.join(selected_outputs)}_{'_'.join(columns_variables)}_{'_'.join(rows_variables)}.csv"
    filename = f"export.csv"

    return Response(csv_string, mimetype='text/csv;charset=utf-8;', headers={'Content-disposition': f'attachment; filename={filename}'})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)