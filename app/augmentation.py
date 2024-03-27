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

# Get breast augmentation data
def get_data():
    with open('static/merged_breast_augmentation.json', 'r') as data_file:
        data = json.load(data_file)

    return data

# Get config
def get_config():
    with open('static/config.json', 'r') as config_file:
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

@app.route('/alldata', methods=['GET'])
def alldata():
    df = get_data()
    if not df:
        return jsonify({"error": "Data not found"}), 404

    # Process fetched data
    data_dict = {}
    max_length = max(len(entry) for entry in df)  # Adjusted this line

    for entry in df:
        for key, value in entry.items():
            data_dict.setdefault(key, []).append(value)

    # Pad shorter lists with None to match the length of the longest list
    for key in data_dict:
        data_dict[key] += [None] * (max_length - len(data_dict[key]))

    return jsonify(data_dict)

@app.route('/datalist', methods=['GET'])
def datalist():
    df = get_data()
    if not df:
        return jsonify({"error": "Data not found"}), 404
    
    # Extract unique keys from all data dictionaries
    unique_keys = set()
    for url_data in df:
        unique_keys.update(url_data.keys())

    # Construct the response JSON containing the list of unique keys
    response_data = {"question": list(unique_keys)}

    # Convert dictionary to JSON and return with headers
    return jsonify(response_data)

@app.route('/data/<selected_data>', methods=['GET'])
def data(selected_data):
    # Split the selected data by comma to get individual column names
    selected_columns = selected_data.split(',')

    df = get_data()
    if not df:
        return jsonify({"error": "Data not found"}), 404

    # Process fetched data based on selected columns
    data_dict = {column: [] for column in selected_columns}
    max_length = 0 

    for row in df:
        for column in selected_columns:
            value = row.get(column)
            if value is not None:
                if isinstance(value, str) and value.isdigit():
                    value = int(value)
            data_dict[column].append(value)
            max_length = max(max_length, len(data_dict[column]))

    # Pad shorter lists with None to match the length of the longest list
    for column in data_dict:
        data_dict[column] += [None] * (max_length - len(data_dict[column]))

    # Convert dictionary to JSON and return
    return jsonify(data_dict)

@app.route('/visualize_data/<selected_data>/<visualization_type>', methods=['GET', 'POST'])
def visualize_data(selected_data, visualization_type):
    selected_column = selected_data.split(',')

    df = get_data()
    if not df:
        return jsonify({"error": "Data not found"}), 404

    # Process fetched data
    data_dict = {}
    max_length = 0  # Track the maximum length of lists

    for row in df:
        for key, value in row.items():
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

    # Convert processed data to a pandas DataFrame
    df = pd.DataFrame(data_dict)

    # Generate the visualization based on the selected type
    plt.figure(figsize=(8, 4))  # Adjust the figure size as needed

    if visualization_type == 'bar':
        sns.countplot(x=selected_column[0], data=df, order=df[selected_column[0]].value_counts().index)
        plt.xlabel(selected_column[0], fontsize=16)  # Use the selected column as xlabel
        plt.ylabel('Count', fontsize=16)
        plt.xticks(rotation=0, ha='right')  # Rotate x-axis labels for better readability
    elif visualization_type == 'line':
        df[selected_column[0]].value_counts().sort_index().plot(kind='line', marker='o')
        plt.xlabel(selected_column[0], fontsize=16)  # Use the selected column as xlabel
        plt.ylabel('Count', fontsize=16)
        plt.xticks(rotation=0, ha='right')  # Rotate x-axis labels for better readability
    elif visualization_type == 'pie':
        value_counts = df[selected_column[0]].value_counts()
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
        plt.xlabel(selected_column[0], fontsize=16)  # Use the selected column as xlabel
    elif visualization_type == 'histogram':
        df[selected_column[0]].plot(kind='hist', bins=10)
        plt.xlabel(selected_column[0], fontsize=16)  # Use the selected column as xlabel
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













def process_crosstabs(records, config):
    data_dict = defaultdict(list)
    max_length = 0
    columns_var_counts = {}
    rows_var_counts = {}

    # Populate data_dict and find max_length
    for record in records:
        for key, value in record.items():
            # Handle None, "null", and empty strings
            if value is None or value == "null" or value == "":
                value = None
            # Convert numeric strings into integers
            elif isinstance(value, str) and value.isdigit():
                value = int(value)

            data_dict[key].append(value)
            max_length = max(max_length, len(data_dict[key]))

    # Pad shorter lists with None to match the length of the longest list
    for key in data_dict:
        data_dict[key] += [None] * (max_length - len(data_dict[key]))

    # Create DataFrame
    df = pd.DataFrame(data_dict)

    results = {}

    # Iterate over columns and rows configurations
    for column_config in config['columns']:
        column_variable = column_config['variable']
        column_name = column_config['name']
        columns_var_counts[column_variable] = df[column_variable].value_counts()

        for row_config in config['rows']:
            row_variable = row_config['variable']
            row_name = row_config['name']
            labels_columns = df[column_variable].map(lambda x: get_element(config, column_variable, x, 'columns')['label'])
            labels_rows = df[row_variable].map(lambda x: get_element(config, row_variable, x, 'rows')['label'])
            rows_var_counts[row_variable] = labels_rows.value_counts().sort_index()  # Sorting counts by index

            # Compute crosstab and percentages
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

            # Calculate chi-square statistic and p-value
            chi2_stat, p_val, _, _ = chi2_contingency(crosstab_result)
            degrees_of_freedom = (crosstab_result.shape[0] - 2) * (crosstab_result.shape[1] - 2)

            # Convert pandas.Series to dictionaries
            rows_var_counts_dict = rows_var_counts[row_variable].to_dict()
            columns_var_counts_dict = columns_var_counts[column_variable].to_dict()

            # Store results
            results[f"{column_name},{row_name}"] = {
                'crosstab': crosstab_result.to_dict(),
                'row_percentage': crosstab_result_row_percent.to_dict(),
                'column_percentage': crosstab_result_column_percent.to_dict(),
                'total_percentage': crosstab_result_total_percent.to_dict(),
                'chi_square_statistic': float(chi2_stat),
                'p_value': float(p_val),
                'degrees_of_freedom': float(degrees_of_freedom),
                'rows_var_counts': rows_var_counts_dict,
                'columns_var_counts': columns_var_counts_dict
            }

    return results


@app.route('/crosstabs', methods=['GET', 'POST'])
def crosstabs():
    records = get_data()

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
        for key, value in record.items():
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
            labels_rows = df[rows_var].apply(lambda x: get_element(config, rows_var, x, 'rows')['label'])
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
    records = get_data()

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