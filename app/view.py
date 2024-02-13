from flask import Flask, render_template
from flask_mysqldb import MySQL
import os

app = Flask(__name__)

# Configure MySQL
app.config['MYSQL_HOST'] = 'host.docker.internal'
app.config['MYSQL_USER'] = 'local'
app.config['MYSQL_PASSWORD'] = 'secret'
app.config['MYSQL_DB'] = 'fluent'

# Initialize MySQL
mysql = MySQL(app)


@app.route('/')
def home():
    # return render_template('index.html')
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM `script_templates`")
    data = cur.fetchall()
    cur.close()
    return str(data)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)