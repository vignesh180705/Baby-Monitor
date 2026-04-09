from models.user_model import create_user, get_user
from database.db import get_connection
from werkzeug.security import generate_password_hash, check_password_hash

def register_user(name, email, password):
    hashed_pw = generate_password_hash(password)
    return create_user(name, email, hashed_pw)

def login_user(email, password):
    user = get_user_by_email(email)
    if user and check_password_hash(user['password'], password):
        return True
    return False
    

def get_user_by_email(email):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE email=?", (email,))
    user = cursor.fetchone()

    conn.close()
    return user