from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session
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
from scipy.stats import f_oneway
import numpy as np
from scipy.stats import pearsonr, spearmanr

app = Flask(__name__)
app.secret_key = 'doms'

# Configure MySQL
app.config['MYSQL_HOST'] = 'host.docker.internal'
app.config['MYSQL_USER'] = 'dan'
app.config['MYSQL_PASSWORD'] = 'dan'
app.config['MYSQL_DB'] = 'fluent'

# Initialize MySQL
mysql = MySQL(app)

def preprocessed_data(mysql_connection, config_path):
    # Fetch data from the database
    cur = mysql_connection.cursor()
    cur.execute("SELECT `url_data` FROM `records` WHERE `survey_code`='lQuDql' AND `status`='cp' AND `test_id`=0")
    data = cur.fetchall()
    cur.close()

    # Convert JSON data to DataFrame
    data_dicts = [json.loads(row[0]) for row in data]
    df = pd.DataFrame(data_dicts)

    # Preprocess data to combine "None" and "null" values
    df = df.apply(lambda col: col.apply(lambda val: None if val == "null" else val))

    # Convert string values to numerical types (int or float)
    df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

    return df

# Function to preprocess data based on the crosstab_config.json file
def preprocess_with_config(data, column, config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    variable_info = next((x for x in config['survey'] if x['variable'] == column), None)
    if variable_info:
        name = variable_info['name']
        data_type = variable_info.get('data_type', None)
        return name, data_type
    else:
        return None, None

# Function to get the name and data type of the selected column based on the config file
def get_column_info(column, config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    variable_info = next((x for x in config['survey'] if x['variable'] == column), None)
    if variable_info:
        name = variable_info['name']
        data_type = variable_info.get('data_type', None)
        return name, data_type
    else:
        return None, None

@app.context_processor
def utility_processor():
    def get_label_for_value(column, value, config_path):
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        # Check if 'survey' key exists in the config
        if 'survey' in config:
            # Search for the variable info in the 'survey' list
            variable_info = next((x for x in config['survey'] if x.get('variable') == column), None)
            if variable_info and 'values' in variable_info:
                # Create a mapping of values to labels
                values_mapping = {item['value']: item['label'] for item in variable_info['values']}
                # Return the label if found, otherwise return the original value
                return values_mapping.get(value, value)
        
        # Return the original value if no mapping found or if 'survey' key is missing
        return value  

    return dict(get_label_for_value=get_label_for_value)

def load_data_and_config():
    # Get preprocessed data and categorical columns
    df = preprocessed_data(mysql.connection, 'static/survey_config.json')

    # Get unique column names
    columns = df.columns.tolist()

    # Load the crosstab configuration file
    with open('static/survey_config.json', 'r') as config_file:
        crosstab_config = json.load(config_file)
    
    # Prepare a dictionary to store column info (name and data type)
    column_info = {}

    # Iterate over columns to get their names and data types
    for column in columns:
        name, data_type = get_column_info(column, 'static/survey_config.json')
        if name is not None:  # Check if the column name is present in the config file
            column_info[column] = {"name": name, "data_type": data_type}
    
    return df, columns, column_info, crosstab_config


# ALL COMPUTATION
def perform_anova(selected_columns, df):
    # Create a list to store the data for each selected column
    data = []

    # Iterate over each selected column and append its data to the 'data' list
    for column in selected_columns:
        data.append(df[column])

    # Perform ANOVA using the collected data
    anova_result = f_oneway(*data)

    # Calculate summary statistics
    summary = df[selected_columns].describe().T
    summary['Sum'] = df[selected_columns].sum()

    return anova_result, summary

def calculate_chi_square(df_selected, column_for_columns, selected_row, total_column_value):
    # Calculate the frequency counts of unique values in selected_row
    selected_row_counts = df_selected[selected_row].value_counts()
    total_selected_row_count = selected_row_counts.sum()

    # Calculate the percentage of each value in selected_row
    selected_row_percentage = (selected_row_counts / total_selected_row_count * 100).astype(float)

    # Calculate the contingency table
    contingency_table = pd.crosstab(df_selected[selected_row], df_selected[column_for_columns])
    
    # Calculate Chi Square statistics
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
    
    return {'Chi Square': chi2, 'p-value': p, 'Degrees of Freedom': dof, 
            'breakdown_table_frequency': breakdown_table, 
            'breakdown_table_percentage': breakdown_table_percentage,
            'selected_row_counts': selected_row_counts,
            'total_selected_row_count': total_selected_row_count,
            'selected_row_percentage': selected_row_percentage}

def calculate_ttest(df_selected, selected_row, column_for_columns):
    # Calculate the frequency counts of unique values in selected_row
    selected_row_counts = df_selected[selected_row].value_counts()
    total_selected_row_count = selected_row_counts.sum()

    # Calculate the percentage of each value in selected_row
    selected_row_percentage = (selected_row_counts / total_selected_row_count * 100).astype(float)
    
    # Calculate the contingency table
    contingency_table = pd.crosstab(df_selected[selected_row], df_selected[column_for_columns])

    # Construct contingency table for t-test
    group1 = df_selected[selected_row]
    group2 = df_selected[column_for_columns]

    # Convert Series to DataFrame
    group1_df = group1.to_frame()
    group2_df = group2.to_frame()

    # Filter out non-numeric values from group1 and group2
    group1_numeric = group1_df.select_dtypes(include=np.number).dropna()
    group2_numeric = group2_df.select_dtypes(include=np.number).dropna()

    # Convert Series to numpy arrays
    group1_array = group1_numeric.to_numpy()
    group2_array = group2_numeric.to_numpy()

    # Calculate mean of each group
    mean_group1 = group1_array.mean()
    mean_group2 = group2_array.mean()

    # Check if the variances of group1 and group2 are equal
    if np.var(group1_array) == np.var(group2_array):
        # Perform equal variance (pooled) t-test
        t_statistic, p_value = scipy.stats.ttest_ind(group1_array, group2_array, equal_var=True)
    else:
        # Check if the total data in group1 and group2 are equal
        if len(group1_array) == len(group2_array):
            # Perform paired t-test
            t_statistic, p_value = scipy.stats.ttest_rel(group1_array, group2_array)
        else:
            # Perform independent t-test (unequal variance t-test)
            t_statistic, p_value = scipy.stats.ttest_ind(group1_array, group2_array, equal_var=False)

    # Separate frequency and percentage in breakdown table
    breakdown_table = contingency_table.copy()
    breakdown_table_percentage = (contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100).astype(float)
    breakdown_percentage = (contingency_table.div(df_selected[column_for_columns].count(), axis=1) * 100).astype(float)

    # Add sum of rows and columns to the breakdown tables
    breakdown_table['Total'] = breakdown_table.sum(axis=1)
    breakdown_table_percentage['Total'] = (breakdown_percentage.sum(axis=1))
    breakdown_table.loc['Total'] = breakdown_table.sum()
    breakdown_table_percentage.loc['Total'] = breakdown_percentage.sum()

    return {'t_statistic': t_statistic, 'p_value': p_value,  
            'breakdown_table_frequency': breakdown_table, 
            'breakdown_table_percentage': breakdown_table_percentage,
            'selected_row_counts': selected_row_counts,
            'total_selected_row_count': total_selected_row_count,
            'selected_row_percentage': selected_row_percentage,
            'mean_group1': mean_group1,'mean_group2': mean_group2}


# Function to compute Pearson correlation coefficient and generate scatterplot
def compute_pearson_correlation(column1, column2, column_info):
    # Calculate Pearson correlation coefficient
    correlation_coefficient, _ = pearsonr(column1, column2)

    # Generate scatterplot
    sns.set(style="whitegrid")  # Set style
    plt.figure(figsize=(8, 6))  # Set figure size
    sns.scatterplot(x=column1, y=column2)  # Create scatterplot
    plt.title('Scatterplot')  # Set title
    plt.xlabel(column_info[column1.name]["name"])  # Set x-axis label using column_info
    plt.ylabel(column_info[column2.name]["name"])  # Set y-axis label using column_info
    scatterplot_path = 'static/scatterplot.png'  # Define file path to save the scatterplot
    plt.savefig(scatterplot_path)  # Save scatterplot to file
    plt.close()  # Close the plot to release memory

    return correlation_coefficient, scatterplot_path

# Function to compute Spearman correlation coefficient and generate scatterplot
def compute_spearman_correlation(column1, column2, column_info):
    # Calculate Spearman correlation coefficient
    correlation_coefficient, _ = spearmanr(column1, column2)

    # Generate scatterplot
    sns.set(style="whitegrid")  # Set style
    plt.figure(figsize=(8, 6))  # Set figure size
    sns.scatterplot(x=column1, y=column2)  # Create scatterplot
    plt.title('Scatterplot')  # Set title
    plt.xlabel(column_info[column1.name]["name"])  # Set x-axis label using column_info
    plt.ylabel(column_info[column2.name]["name"])  # Set y-axis label using column_info
    scatterplot_path = 'static/scatterplot.png'  # Define file path to save the scatterplot
    plt.savefig(scatterplot_path)  # Save scatterplot to file
    plt.close()  # Close the plot to release memory

    return correlation_coefficient, scatterplot_path

@app.route('/')
def home():
    # Pass the column names to the template
    return render_template('dashboard.html')

@app.route('/visualize')
def visualize():
    # Load data and configuration
    df, columns, column_info, crosstab_config = load_data_and_config()

    # Pass necessary data to the template
    return render_template('visualize.html', columns=columns, column_info=column_info, crosstab_config=crosstab_config)

@app.route('/visualize_data', methods=['POST'])
def visualize_data():
    # Get the selected column name and visualization type from the form
    selected_column = request.form['column']
    visualization_type = request.form['visualization']

    # Load data and configuration
    df, columns, column_info, crosstab_config = load_data_and_config()

    # Get the name and data type for the selected column based on the config file
    column_name, column_data_type = get_column_info(selected_column, 'static/survey_config.json')

    # Ensure the data in the selected column is numeric
    df[selected_column] = pd.to_numeric(df[selected_column], errors='coerce')

    # Generate the visualization based on the selected type
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

    if visualization_type == 'bar':
        sns.countplot(x=selected_column, data=df, order=df[selected_column].value_counts().index)
        for i, v in enumerate(df[selected_column].value_counts()):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.xlabel(column_name, fontsize=16)  # Use the name from the config file
        plt.ylabel('Count', fontsize=16)
        plt.xticks(rotation=0, ha='right')  # Rotate x-axis labels for better readability
    elif visualization_type == 'line':
        df[selected_column].value_counts().sort_index().plot(kind='line', marker='o')
        plt.xlabel(column_name, fontsize=16)  # Use the name from the config file
        plt.ylabel('Count', fontsize=16)
        plt.xticks(rotation=0, ha='right')  # Rotate x-axis labels for better readability
    elif visualization_type == 'pie':
        value_counts = df[selected_column].value_counts()
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
        plt.xlabel(column_name, fontsize=16)  # Use the name from the config file
    elif visualization_type == 'histogram':
        df[selected_column].plot(kind='hist', bins=10)
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

    # Load the crosstab configuration file
    with open('static/crosstab_config.json', 'r') as config_file:
        crosstab_config = json.load(config_file)

    # Calculate column data (value, count, percentage) for the selected column
    if selected_column in columns:
        column_data = df[selected_column].value_counts().reset_index()
        column_data.columns = ['Value', 'Count']
        total_count = column_data['Count'].sum()
        column_data['Percentage'] = (column_data['Count'] / total_count) * 100
        column_data = column_data.values.tolist()
    else:
        column_data = None

    # Pass the visualization URI, column name, data type, table data, and other data to the template
    return render_template('visualize.html', visualization=visualization_uri, columns=columns, column_info=column_info, column_data=column_data, column_name=column_name, column_data_type=column_data_type, selected_column=selected_column, crosstab_config=crosstab_config)

@app.route('/chisquare')
def chisquare():
    # Load data and configuration
    df, columns, column_info, crosstab_config = load_data_and_config()

    # Pass the column names to the template
    return render_template('chisquare.html', columns=columns, column_info=column_info, crosstab_config=crosstab_config)

@app.route('/compute_chisquare', methods=['POST'])
def compute_chisquare():
    # Get selected column names and computation method from the form
    column_for_columns = request.form['column_for_columns']
    column_for_rows = request.form.getlist('column_for_rows')  # Get list of selected rows
    computation_method = request.form['computation_method']

    # Load data and configuration
    df, columns, column_info, crosstab_config = load_data_and_config()

    # Filter DataFrame based on selected columns
    df_selected = df[[column_for_columns] + column_for_rows]

    # Calculate the total value of the selected column
    total_column_value = df[column_for_columns].count()

    # Perform computation based on the selected method
    result = {}
    if computation_method == 'chi_square':
        for selected_row in column_for_rows:
            # Calculate Chi Square statistics
            result[selected_row] = calculate_chi_square(df_selected, column_for_columns, selected_row, total_column_value)

    # Calculate maximum frequency
    maximum_frequency = df[column_for_columns].max()

    # Pass the result and selected columns to the template
    return render_template('chisquare.html', crosstab_config=crosstab_config, maximum_frequency=maximum_frequency, column_info=column_info, total_column_value=total_column_value, result=result, columns=columns, column_for_columns=column_for_columns, column_for_rows_list=column_for_rows, computation_method=computation_method)

@app.route('/ttest')
def ttest():
    # Load data and configuration
    df, columns, column_info, crosstab_config = load_data_and_config()

    # Pass the column names to the template
    return render_template('Ttest.html', columns=columns, column_info=column_info, crosstab_config=crosstab_config)

@app.route('/compute_ttest', methods=['POST'])
def compute_ttest():
    # Get selected column names and computation method from the form
    column_for_columns = request.form['column_for_columns']
    column_for_rows = request.form.getlist('column_for_rows')  # Get list of selected rows
    computation_method = request.form['computation_method']

    # Load data and configuration
    df, columns, column_info, crosstab_config = load_data_and_config()

    # Filter DataFrame based on selected columns
    df_selected = df[[column_for_columns] + column_for_rows]

    # Perform computation based on the selected method
    result = {}
    if computation_method == 'ttest':
        for selected_row in column_for_rows:
            # Calculate t-test statistics
            result[selected_row] = calculate_ttest(df_selected, selected_row, column_for_columns)

    # Calculate maximum frequency
    maximum_frequency = df[column_for_columns].max()

    # Pass the result and selected columns to the template
    return render_template('Ttest.html',  mean_group1=result[column_for_rows[0]]['mean_group1'], mean_group2=result[column_for_rows[0]]['mean_group2'], crosstab_config=crosstab_config, maximum_frequency=maximum_frequency, column_info=column_info, result=result, columns=columns, column_for_columns=column_for_columns, column_for_rows_list=column_for_rows, computation_method=computation_method)


@app.route('/correlation')
def correlation():
    # Load data and configuration
    df, columns, column_info, crosstab_config = load_data_and_config()

    # Pass the column names to the template
    return render_template('correlation.html', columns=columns, column_info=column_info, crosstab_config=crosstab_config)

@app.route('/compute_correlation', methods=['POST'])
def compute_correlation():
    # Get selected column names and computation method from the form
    column_for_columns = request.form['column_for_columns']
    column_for_rows = request.form.getlist('column_for_rows')  # Get list of selected rows
    computation_method = request.form['computation_method']

    # Load data and configuration
    df, columns, column_info, crosstab_config = load_data_and_config()

    # Filter DataFrame based on selected columns
    df_selected = df[[column_for_columns] + column_for_rows]

    # Calculate the total value of the selected column
    total_column_value = df[column_for_columns].count()

    # Perform computation based on the selected method
    result = {}
    if computation_method == 'correlation':
        # Perform Pearson correlation computation
        result = {}
        for selected_row in column_for_rows:
            # Calculate Pearson correlation coefficient and scatterplot
            correlation_coefficient, scatterplot_path = compute_pearson_correlation(df_selected[column_for_columns], df_selected[selected_row], column_info)

            # Add result to the dictionary
            result[selected_row] = {'correlation_coefficient': correlation_coefficient, 'scatterplot_path': scatterplot_path}
    elif computation_method == 'spearman':
        # Perform Spearman correlation computation
        result = {}
        for selected_row in column_for_rows:
            # Calculate Spearman correlation coefficient and scatterplot
            correlation_coefficient, scatterplot_path = compute_spearman_correlation(df_selected[column_for_columns], df_selected[selected_row], column_info)

            # Add result to the dictionary
            result[selected_row] = {'correlation_coefficient': correlation_coefficient, 'scatterplot_path': scatterplot_path}

    # Pass the result and selected columns to the template
    return render_template('correlation.html', crosstab_config=crosstab_config, column_info=column_info, result=result, columns=columns, column_for_columns=column_for_columns, column_for_rows_list=column_for_rows, computation_method=computation_method)

@app.route('/anova')
def anova():
    # Load data and configuration
    df, columns, column_info, crosstab_config = load_data_and_config()

    # Pass the column names to the template
    return render_template('anova.html', columns=columns, column_info=column_info, crosstab_config=crosstab_config)

@app.route('/compute_anova', methods=['POST'])
def compute_anova():
    # Select specific columns for ANOVA computation
    selected_columns = request.form.getlist('columns')

    # Load data and configuration
    df, columns, column_info, crosstab_config = load_data_and_config()
    
    # Perform ANOVA
    anova_result, summary = perform_anova(selected_columns, df)

    return render_template('anova.html', anova_result=anova_result, summary=summary, columns=columns, column_info=column_info)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)