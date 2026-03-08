import mysql.connector

db_config = {
    'user': 'root',
    'password': '1234567890', 
    'host': '127.0.0.1',
    'database': 'ckd_db',
    'auth_plugin': 'mysql_native_password'
}

try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

   
    cursor.execute("SHOW TABLES;")
    tables = cursor.fetchall()
    print("Tables in ckd_db:", tables)

   
    cursor.execute("DESCRIBE patients;")
    columns = cursor.fetchall()
    print("patients table structure:")
    for col in columns:
        print(col)

    conn.close()

except mysql.connector.Error as e:
    print("Error:", e)