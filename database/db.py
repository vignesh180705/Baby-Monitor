import sqlite3
from config import DATABASE

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS audio_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            predicted_label TEXT,
            non_cry REAL,
            cry REAL
        )
    ''')

    conn.commit()
    conn.close()

def get_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  
    return conn