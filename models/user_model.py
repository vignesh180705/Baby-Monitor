from database.db import get_connection

def create_user(name, email, password):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email, password)
        )
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()


def get_user(email, password):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE email=? AND password=?",
        (email, password)
    )
    user = cur.fetchone()
    conn.close()

    return user