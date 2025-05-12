import pymysql

try:
    connection = pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="login-register",
        port=3307
    )
except pymysql.err.OperationalError as e:
    print(f"Error: {e}")
