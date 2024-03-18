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
def process_crosstabs(rows, independent_variables, dependent_variables):
    data_dict = {}
    max_length = 0

    for row in rows:
        url_data = json.loads(row[0])
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
    for independent_var in independent_variables:
        labels_independent = df[independent_var].apply(lambda x: get_label(independent_var, x, 'static/survey_config.json'))
        for dependent_var in dependent_variables:
            labels_dependent = df[dependent_var].apply(lambda x: get_label(dependent_var, process_value(x), 'static/survey_config.json'))

            crosstab_result = pd.crosstab(labels_independent, labels_dependent, margins=True, margins_name='Total')
            crosstab_result_row_percent = pd.crosstab(labels_independent, labels_dependent, normalize='index')
            crosstab_result_column_percent = pd.crosstab(labels_independent, labels_dependent, normalize='columns')
            crosstab_result_total_percent = pd.crosstab(labels_independent, labels_dependent, normalize='all', margins=True, margins_name='Total')

            # Calculate row totals
            row_totals = crosstab_result_row_percent.sum(axis=1)
            total_row = pd.DataFrame({'Total': [1] * len(row_totals)}, index=row_totals.index)
            crosstab_result_row_percent = pd.concat([crosstab_result_row_percent, total_row], axis=1)

            # Calculate column totals
            column_totals = crosstab_result_column_percent.sum()
            total_column = pd.DataFrame({'Total': [1] * len(column_totals)}, index=column_totals.index).transpose()
            crosstab_result_column_percent = pd.concat([crosstab_result_column_percent, total_column])

            chi2_stat, p_val, dof, expected = chi2_contingency(crosstab_result)

            results[f"{independent_var},{dependent_var}"] = {
                'crosstab': crosstab_result.to_dict(),
                'row_percentage': crosstab_result_row_percent.to_dict(),
                'column_percentage': crosstab_result_column_percent.to_dict(),
                'total_percentage': crosstab_result_total_percent.to_dict(),
                'chi_square_statistic': float(chi2_stat),
                'degrees_of_freedom': int(dof),
                'p_value': float(p_val)
            }

    return results


@app.route('/crosstabs/<independent>/<dependent>', methods=['GET', 'POST'])
def crosstabs(independent, dependent):
    dependent_variables = dependent.split(',')
    independent_variables = independent.split(',')
    cur = mysql.connection.cursor()
    cur.execute("SELECT url_data FROM records WHERE survey_code='lQuDql' AND status='cp' AND test_id=0")
    rows = cur.fetchall()
    cur.close()

    results = process_crosstabs(rows, independent_variables, dependent_variables)

    # Convert dictionary to JSON string
    json_string = json.dumps(results)

    # Return JSON string with proper content type
    return Response(json_string, content_type='application/json')


# Export to CSV API
def process_crosstabs_csv(rows, independent_variables, dependent_variables):
    data_dict = {}
    max_length = 0  
    
    for row in rows:
        url_data = json.loads(row[0])
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
    independent_var_counts = {}
    dependent_var_counts = {}
    dependent_var_percentage = {}
    for independent_var in independent_variables:
        independent_var_counts[independent_var] = df[independent_var].value_counts()
        
        labels_independent = df[independent_var].apply(lambda x: get_label(independent_var, x, 'static/survey_config.json'))
        for dependent_var in dependent_variables:
            labels_dependent = df[dependent_var].apply(lambda x: get_label(dependent_var, process_value(x), 'static/survey_config.json'))
            dependent_var_counts[dependent_var] = labels_dependent.value_counts().sort_index()  # Sorting counts by index
            dependent_var_counts[dependent_var]['Total'] = df[dependent_var].value_counts().sum()
            dependent_var_percentage[dependent_var] = ((dependent_var_counts[dependent_var]) / len(df[dependent_var])) * 100

            crosstab_result = pd.crosstab(labels_dependent, labels_independent, margins=True, margins_name='Total')
            crosstab_result_row_percent = pd.crosstab(labels_dependent, labels_independent, normalize='index')
            crosstab_result_column_percent = pd.crosstab(labels_dependent, labels_independent, normalize='columns')
            crosstab_result_total_percent = pd.crosstab(labels_dependent, labels_independent, normalize='all', margins=True, margins_name='Total')

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

            if independent_var not in results:
                results[independent_var] = {}
            results[independent_var][dependent_var] = {
                'crosstab': crosstab_result,
                'row_percentage': crosstab_result_row_percent,
                'column_percentage': crosstab_result_column_percent,
                'total_percentage': crosstab_result_total_percent,
                'chi_square_statistic': chi2_stat,
                'degrees_of_freedom': degrees_of_freedom,
                'p_value': p_val
            }

    return results, independent_var_counts, dependent_var_counts, dependent_var_percentage

@app.route('/export_csv/<independent>/<dependent>/<csv_outputs>', methods=['GET'])
def export_crosstabs_csv(independent, dependent, csv_outputs):
    independent_variables = independent.split(',')
    dependent_variables = dependent.split(',')
    cur = mysql.connection.cursor()
    cur.execute("SELECT url_data FROM records WHERE survey_code='lQuDql' AND status='cp' AND test_id=0")
    rows = cur.fetchall()
    cur.close()

    results, independent_var_counts, dependent_var_counts, dependent_var_percentage = process_crosstabs_csv(rows, independent_variables, dependent_variables)

    # Check if csv_output parameter is valid
    valid_outputs = ['frequency_crosstab', 'row_percentage', 'column_percentage', 'total_percentage', 'total_crosstab', 'chi_square_results', 'row_crosstab', 'column_crosstab']
    selected_outputs = csv_outputs.split(',')
    for output in selected_outputs:
        if output not in valid_outputs:
            return f"Invalid csv_output parameter: {output}", 400

    # Export selected outputs to CSV
    csv_data = []

    for independent_var in independent_variables:
        for dependent_var in dependent_variables:
            # Retrieve names of independent and dependent variables
            independent_name, _ = get_column_info(independent_var, 'static/survey_config.json')
            dependent_name, _ = get_column_info(dependent_var, 'static/survey_config.json')

            for csv_output in selected_outputs:
                if csv_output == 'frequency_crosstab':
                    # Add crosstab data
                    crosstab = results[independent_var][dependent_var]['crosstab']
                    dependent_labels = [f"{dependent_name}"] + [''] + [''] + [''] + [f'"{label}"' for label in crosstab.columns.tolist()]

                    csv_data.append([''] + [''] + [f"All"] + [''] + [''] + [f"{independent_name}"])
                    csv_data.append(["Crosstab"])
                    csv_data.append(dependent_labels)
                    dep_values = dependent_var_counts[dependent_var]
                    # Iterate over crosstab rows and dependent_var_counts simultaneously
                    for (index, row), dep in zip(crosstab.iterrows(), dep_values):
                        csv_data.append([f'"{index}"'] + ['Frequency'] + [dep] + ['-'] + row.values.tolist())
                    csv_data.append([''])

                elif csv_output in ['row_percentage', 'column_percentage', 'total_percentage']:
                    percentage_type = csv_output.split('_')[0].capitalize()
                    percentage_data = results[independent_var][dependent_var][csv_output]
                    dependent_labels = [f"{dependent_name}"] + [''] + [''] + [''] + [f'"{label}"' for label in percentage_data.columns.tolist()]
                    
                    csv_data.append([''] + [''] + ['All'] + [''] + [''] + [f"{independent_name}"])
                    csv_data.append([f"{percentage_type} Percentage"])
                    csv_data.append(dependent_labels)
                    dep_percent = dependent_var_percentage[dependent_var]
                    for (index, row), percent in zip(percentage_data.iterrows(), dep_percent):
                        if csv_output == 'column_percentage':
                            csv_data.append([f'"{index}"'] + ['Column%'] + [f'{percent:.2f}%'] + ['-'] + [f"{value*100:.2f}%" for value in row.values.tolist()])
                        elif csv_output == 'row_percentage':
                            csv_data.append([f'"{index}"'] + ['Row%'] + [f'{percent:.2f}%'] + ['-'] + [f"{value*100:.2f}%" for value in row.values.tolist()])
                        elif csv_output == 'total_percentage':
                            csv_data.append([f'"{index}"'] + ['Total%'] + [f'{percent:.2f}%'] + ['-'] + [f"{value*100:.2f}%" for value in row.values.tolist()])
                    csv_data.append([''])
                        
                elif csv_output == 'chi_square_results':
                    csv_data.append(["Chi-Square Result"])
                    csv_data.append([f"{dependent_name} vs {independent_name}"])
                    p_value = results[independent_var][dependent_var]['p_value']
                    chi_square_statistic = results[independent_var][dependent_var]['chi_square_statistic']
                    degrees_of_freedom = results[independent_var][dependent_var]['degrees_of_freedom']
                    significance = "<" if p_value < 0.05 else ">"
                    significance_text = "significant" if p_value < 0.05 else "non-significant"
                    total_values = (independent_var_counts[independent_var]).sum()
                    csv_data.append([f'"The association between {dependent_name} and which best describes your {independent_name}? is {significance_text} X\u00B2 ({total_values}) = {chi_square_statistic:.2f}, df = {degrees_of_freedom}, p {significance} 0.5"'])
                    csv_data.append([''])
                
                elif csv_output in ['row_crosstab', 'column_crosstab', 'total_crosstab']:
                    if csv_output == 'total_crosstab':
                        csv_data.append(["Crosstabs and Total Percentage Data"])
                    elif csv_output == 'row_crosstab':
                        csv_data.append(["Crosstabs and Row Percentage Data"])
                    elif csv_output == 'column_crosstab':
                        csv_data.append(["Crosstabs and Column Percentage Data"])

                    csv_data.append([''] + [''] + ['All'] + [''] + [''] + [f"{independent_name}"])
                    crosstab = results[independent_var][dependent_var]['crosstab']
                    crosstab_iterator = crosstab.iterrows()
                    dep_values = dependent_var_counts[dependent_var]
                    dep_percent = dependent_var_percentage[dependent_var]
                    p_value = results[independent_var][dependent_var]['p_value']
                    chi_square_statistic = results[independent_var][dependent_var]['chi_square_statistic']
                    degrees_of_freedom = results[independent_var][dependent_var]['degrees_of_freedom']
                    significance = "<" if p_value < 0.05 else ">"
                    significance_text = "significant" if p_value < 0.05 else "non-significant"
                    total_values = (independent_var_counts[independent_var]).sum()
                    dependent_labels = [f"{dependent_name}"] + [''] + [''] + [''] + [f'"{label}"' for label in crosstab.columns.tolist()]
                    csv_data.append(dependent_labels)

                    if csv_output == 'total_crosstab':
                        total_percentage = results[independent_var][dependent_var]['total_percentage']
                        total_percentage_iterator = total_percentage.iterrows()
                        for (index, (crosstab_index, crosstab_row)), dep, (total_percentage_index, total_percentage_row), percent in zip(enumerate(crosstab_iterator), dep_values, total_percentage_iterator, dep_percent):
                            csv_data.append([f'"{crosstab_index}"'] + ['Frequency'] + [dep] + ['-'] + crosstab_row.values.tolist())
                            csv_data.append([''] + ['Total%'] + [f'{percent:.2f}%'] + ['-'] + [f'"{(value*100):.2f}%"' for value in total_percentage_row.tolist()])
                    elif csv_output == 'row_crosstab':
                        row_percentage = results[independent_var][dependent_var]['row_percentage']
                        row_percentage_iterator = row_percentage.iterrows()
                        for (index, (crosstab_index, crosstab_row)), dep, (row_percentage_index, row_percentage_row), percent in zip(enumerate(crosstab_iterator), dep_values, row_percentage_iterator, dep_percent):
                            csv_data.append([f'"{crosstab_index}"'] + ['Frequency'] + [dep] + ['-'] + crosstab_row.values.tolist())
                            csv_data.append([''] + ['Row%'] + [f'{percent:.2f}%'] + ['-'] + [f'"{(value*100):.2f}%"' for value in row_percentage_row.tolist()])
                    elif csv_output == 'column_crosstab':
                        column_percentage = results[independent_var][dependent_var]['column_percentage']
                        column_percentage_iterator = column_percentage.iterrows()
                        for (index, (crosstab_index, crosstab_row)), dep, (column_percentage_index, column_percentage_row), percent in zip(enumerate(crosstab_iterator), dep_values, column_percentage_iterator, dep_percent):
                            csv_data.append([f'"{crosstab_index}"'] + ['Frequency'] + [dep] + ['-'] + crosstab_row.values.tolist())
                            csv_data.append([''] + ['Column%'] + [f'{percent:.2f}%'] + ['-'] + [f'"{(value*100):.2f}%"' for value in column_percentage_row.tolist()])

                    csv_data.append([f'"The association between {dependent_name} and which best describes your {independent_name}? is {significance_text} X\u00B2 ({total_values}) = {chi_square_statistic:.2f}, df = {degrees_of_freedom}, p {significance} 0.5"'])
                    csv_data.append([''])

            # Add empty line for separation between dependent variables
            csv_data.append([""])

        # Add empty line for separation between independent variables
        csv_data.append([""])

    # Generate CSV string
    csv_string = "\ufeff" + "\n".join([",".join(map(str, row)) for row in csv_data])

    # Set filename based on selected csv_outputs
    filename = f"{'_'.join(selected_outputs)}_{'_'.join(independent_variables)}_{'_'.join(dependent_variables)}.csv"

    return Response(csv_string, mimetype='text/csv;charset=utf-8;', headers={'Content-disposition': f'attachment; filename={filename}'})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)