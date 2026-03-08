import mysql.connector


db_config = {
    'user': 'root',
    'password': '1234567890', 
    'host': '127.0.0.1'
}


conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()


cursor.execute("CREATE DATABASE IF NOT EXISTS ckd_db")
print("Database ckd_db created or already exists ✅")

cursor.close()
conn.close()