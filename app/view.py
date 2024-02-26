from flask import Flask, render_template, request, send_file
from flask_mysqldb import MySQL
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import chi2_contingency
from io import BytesIO

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
    cur.execute("SELECT `url_data` FROM `records` WHERE `survey_code`='lQuDql' AND `status`='cp' AND `test_id`=0 LIMIT 10")
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


@app.route('/test')
def test():
    # Fetch data from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT `url_data` FROM `records` WHERE `survey_code`='lQuDql' AND `status`='cp' AND `test_id`=0")
    data = cur.fetchall()
    cur.close()

    cur2 = mysql.connection.cursor()
    cur2.execute("SELECT `config` FROM `cross-tabs` WHERE `survey_id`=414")
    data2 = cur2.fetchone()
    cur2.close()

    crosstab = json.loads(data2[0])
    rows = crosstab['rows']

    return render_template('dump.html', variable=json.dumps(rows))
    filtered_data = []

    for row in rows:
        for entry in data:
            field_entry = json.loads(entry[0])
            filtered_entry = {}
            for setting in row:
                return render_template('dump.html', variable=json.dumps(setting))
                variable = setting['variable']
                values = [str(item['value']) for item in row['values']]
                if variable in field_entry and str(field_entry[variable]) in values:
                    filtered_entry[variable] = field_entry[variable]
            filtered_data.append(filtered_entry)
    return render_template('dump.html', variable=json.dumps(filtered_data))

    # Convert JSON data to DataFrame
    parsed_data = []
    for json_string in data:
        parsed_json = {key: value for key, value in json.loads(json_string[0]).items() if value is not None and value != 'null'}
    # Convert remaining values to strings
        parsed_json_str = {key: str(value) for key, value in parsed_json.items()}
        parsed_data.append(parsed_json_str)

    df = pd.DataFrame(parsed_data)

    # Specify column names for chi-square test
    column1 = 'Age_group'
    column2 = 'a3'

    desired_columns = [column1, column2]
    counts_table = get_counts_table(df, desired_columns)

    # # Create a contingency table (counts)
    contingency_table = pd.crosstab(df[column1], df[column2])

    contingency_table_percentage = pd.crosstab(df[column1], df[column2], margins=True, normalize='index') * 100
    contingency_table_percentage_html = contingency_table_percentage.to_html()

    contingency_table_html = contingency_table.to_html()

    # # Perform chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)

    # excel_buffer = BytesIO()
    # contingency_table.to_excel(excel_buffer, index=True)
    # excel_buffer.seek(0)
    # return send_file(excel_buffer, attachment_filename='cross_tabulation.xlsx', as_attachment=True)

    return render_template('test2.html', df=df, chi2=chi2, p=p, dof=dof, expected_df=expected_df, contingency_table=contingency_table, counts_table=counts_table, contingency_table_html=contingency_table_html, contingency_table_percentage_html=contingency_table_percentage_html)

# Function to get counts table
def get_counts_table(df, columns):
    counts = {}
    for column in columns:
        counts[column] = df[column].dropna().astype(str).value_counts().to_dict()
    return counts

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
