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


app = Flask(__name__)
app.secret_key = 'doms'

# Configure MySQL
app.config['MYSQL_HOST'] = 'host.docker.internal'
app.config['MYSQL_USER'] = 'dan'
app.config['MYSQL_PASSWORD'] = 'dan'
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
    try:
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
        plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

        if visualization_type == 'bar':
            sns.countplot(x=selected_column, data=df, order=df[selected_column].value_counts().index)
            for i, v in enumerate(df[selected_column].value_counts()):
                plt.text(i, v + 0.5, str(v), ha='center')
            plt.xlabel(column_label, fontsize=16)  # Use the label from the config file
            plt.ylabel('Count', fontsize=16)
            plt.xticks(rotation=10, ha='right')  # Rotate x-axis labels for better readability
        elif visualization_type == 'line':
            df[selected_column].value_counts().sort_index().plot(kind='line', marker='o')
            plt.xlabel(column_label, fontsize=16)  # Use the label from the config file
            plt.ylabel('Count', fontsize=16)
            plt.xticks(rotation=10, ha='right')  # Rotate x-axis labels for better readability
        elif visualization_type == 'pie':
            value_counts = df[selected_column].value_counts()
            plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%')
            plt.xlabel(column_label, fontsize=16)  # Use the label from the config file
        elif visualization_type == 'histogram':
            df[selected_column].plot(kind='hist', bins=10)
            plt.xlabel(column_label, fontsize=16)  # Use the label from the config file
            plt.ylabel('Frequency', fontsize=16)
            plt.xticks(rotation=10, ha='right')  # Rotate x-axis labels for better readability

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
    except Exception as e:
        error_message = 'Selected column does not contain numeric data'
        flash(f'Error: {error_message}', 'error')  # Flash the error message
        print(f'Flashed message: {error_message}')  # Print the flashed message for debugging
        return redirect(url_for('index'))  # Redirect back to the index page


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

        # Load the crosstab configuration file
        with open('static/crosstab_config.json', 'r') as config_file:
            crosstab_config = json.load(config_file)

        # Pass the result and selected columns to the template
        return render_template('crosstabs.html', crosstab_config=crosstab_config, selected_row_percentage_sum=selected_row_percentage_sum, total_selected_row_count=total_selected_row_count, total_column_value=total_column_value, result=result, columns=categorical_columns, column_for_columns=column_for_columns, column_for_rows_list=column_for_rows, computation_method=computation_method)
    except ValueError as e:
        error_message = 'An error occurred: ' + str(e)
        return render_template('crosstabs.html', error_message=error_message, crosstab_config=crosstab_config, columns=categorical_columns)

@app.route('/anova', methods=['GET'])
def anova():
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
    return render_template('anova.html', columns=categorical_columns)


@app.route('/anova_result', methods=['POST', 'GET'])
def anova_result():
    if request.method == 'POST':
        # Fetch data from the database
        cur = mysql.connection.cursor()
        cur.execute("SELECT `url_data` FROM `records`")
        data = cur.fetchall()
        cur.close()

        # Convert JSON data to DataFrame
        data_dicts = [json.loads(row[0]) for row in data]
        df = pd.DataFrame(data_dicts)

        # Select specific columns for ANOVA computation
        selected_columns = request.form.getlist('columns')

        # Preprocess data to combine "None" and "null" values
        for column in df.columns:
            df[column] = preprocess_data(df[column])

        # Convert string values to numeric types (int or float)
        df = df.apply(convert_to_numeric)

        # Compute ANOVA
        anova_result = f_oneway(df[selected_columns[0]], df[selected_columns[1]], df[selected_columns[2]])

        # Calculate summary statistics
        summary = df[selected_columns].describe().T
        summary['Sum'] = df[selected_columns].sum() 

        return render_template('anova_result.html', anova_result=anova_result, summary=summary)

    else:
        # Handle GET request (if necessary)
        return "Please submit the form."

@app.route('/Ttest')
def Ttest():
        # Fetch data from the database
        cur = mysql.connection.cursor()
        cur.execute("SELECT url_data FROM records")
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
        cur.execute("SELECT url_data FROM records")
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
                t_statistic, p_value = scipy.stats.ttest_ind(group1, group2, equal_var=True)
                
                # Separate frequency and percentage in breakdown table
                breakdown_table = contingency_table.copy()
                breakdown_table_percentage = (contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100).astype(float)
                breakdown_percentage = (contingency_table.div(total_column_value, axis=1) * 100).astype(float)

                head_cell_count = 20  # Set a default value for head_cell_count
 
                # Check if selected_row is iterable (e.g., a list)
                if hasattr(selected_row, '_iter_'):
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

@app.route('/correlations')
def correlations():
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

    # Convert string values to numerical types, if not already done
    df = df.apply(convert_to_numeric)

    # Pass numerical columns to the template for the user to select from
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    return render_template('correlations.html', columns=numerical_columns)

def generate_scatter_plot_base64(df, x_column, y_column, pearson_correlation=None):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x=x_column, y=y_column)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Scatter Plot of {x_column} vs {y_column}')        
    plt.text(23, -9, f'Pearson Correlation Coefficient: { pearson_correlation }', fontsize=15, color='black')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64


@app.route('/compute_correlation', methods=['POST'])
def compute_correlation():
    # Get selected columns from the form
    x_column = request.form.get('x_column')
    y_column = request.form.get('y_column')

    # Fetch and preprocess the data again
    cur = mysql.connection.cursor()
    cur.execute("SELECT `url_data` FROM `records`")
    data = cur.fetchall()
    cur.close()

    # Convert JSON data to DataFrame and preprocess
    data_dicts = [json.loads(row[0]) for row in data]
    df = pd.DataFrame(data_dicts)
    for column in df.columns:
        df[column] = preprocess_data(df[column])
    df = df.apply(convert_to_numeric)

    scatter_plot_base64 = None
    pearson_correlation = None

    if x_column and y_column:
        # Ensure columns are numeric for Pearson correlation
        if df[x_column].dtype in ['float64', 'int64'] and df[y_column].dtype in ['float64', 'int64']:
            # Compute the Pearson correlation coefficient
            correlation, p_value = scipy.stats.pearsonr(df[x_column].dropna(), df[y_column].dropna())
            pearson_correlation = f"{correlation:.3f}"

            # Generate scatter plot and encode to base64
            scatter_plot_base64 = generate_scatter_plot_base64(df, x_column, y_column, pearson_correlation)

    # Re-fetch the numerical columns for form repopulation
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Pass the base64 encoded image, Pearson correlation, column selections, and numerical columns back to the template
    return render_template('correlations.html', scatter_plot=scatter_plot_base64, pearson_correlation=pearson_correlation, x_column=x_column, y_column=y_column, columns=numerical_columns)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
