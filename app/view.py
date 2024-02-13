from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
app = Flask(__name__)

# Configure MySQL
app.config['MYSQL_HOST'] = 'host.docker.internal'
app.config['MYSQL_USER'] = 'local'
app.config['MYSQL_PASSWORD'] = 'secret'
app.config['MYSQL_DB'] = 'fluent'

# Initialize MySQL
mysql = MySQL(app)

# Function to preprocess data and combine "None" and "null" values
def preprocess_data(data):
    return [val if val not in ['None', 'null'] else None for val in data]

@app.route('/')
def home():
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

    # Get unique column names
    columns = df.columns.tolist()

    # Pass the column names to the template
    return render_template('index.html', columns=columns)


@app.route('/column_data', methods=['POST'])
def column_data():
    # Get the selected column name from the form
    selected_column = request.form['column']

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

    # Get unique values from the selected column
    column_data = df[selected_column].value_counts()

    # Combine "None" and "null" values
    if None in column_data.index:
        column_data['None/null'] = column_data[None]
        column_data.drop(None, inplace=True)

    # Calculate percentages
    total_count = column_data.sum()
    column_data_percentage = (column_data / total_count) * 100

    # Create a bar plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=column_data.index, y=column_data)
    plt.title(f'Counts of Unique Values for {selected_column}')
    plt.xlabel(selected_column)
    plt.ylabel('Count')

    # Add percentages to the bars if data exists
    if not column_data.empty:
        for i, v in enumerate(column_data):
            plt.text(i, v + 0.5, f'{v} ({column_data_percentage[i]:.2f}%)', ha='center', va='bottom')

    # Save the plot as a file
    plot_path = 'static/plot.png'
    plt.savefig(plot_path)

    # Get unique column names
    columns = df.columns.tolist()

    # Pass the plot path and column names to the template
    return render_template('index.html', plot_path=plot_path, selected_column=selected_column, columns=columns)

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

    # Get unique column names
    columns = df.columns.tolist()

    # Pass the column names to the template
    return render_template('crosstabs.html', columns=columns)

@app.route('/compute_crosstab', methods=['POST'])
def compute_crosstab():
    # Get selected column names and computation method from the form
    column_for_columns = request.form['column_for_columns']
    column_for_rows = request.form['column_for_rows']
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

    # Filter DataFrame based on selected columns
    df_selected = df[[column_for_columns, column_for_rows]]

    # Perform computation based on the selected method
    if computation_method == 'chi_square':
        # Perform Chi Square computation
        contingency_table = pd.crosstab(df_selected[column_for_rows], df_selected[column_for_columns])
        chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency_table)
        result = {'Chi Square': chi2, 'p-value': p, 'Degrees of Freedom': dof}
    else:
        result = "Invalid computation method"

    # Pass the result and selected columns to the template
    return render_template('crosstabs.html', result=result, column_for_columns=column_for_columns, column_for_rows=column_for_rows, computation_method=computation_method)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
