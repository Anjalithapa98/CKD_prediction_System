import mysql.connector

# MySQL connection config
db_config = {
    'user': 'root',
    'password': '1234567890',  # your MySQL root password
    'host': '127.0.0.1'
}

# Connect to MySQL
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# Create database
cursor.execute("CREATE DATABASE IF NOT EXISTS ckd_db")
print("Database ckd_db created or already exists ✅")

cursor.close()
conn.close()