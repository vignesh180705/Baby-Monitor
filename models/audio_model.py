from database.db import get_connection

def get_audio_result(filename):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM audio_results WHERE filename=?", (filename,))
    result = cur.fetchone()
    conn.close()

    return result


def insert_audio_result(filename, predicted_label, non_cry, cry):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute('''
        INSERT INTO audio_results (filename, predicted_label, non_cry, cry)
        VALUES (?, ?, ?, ?)
    ''', (filename, predicted_label, non_cry, cry))

    conn.commit()
    conn.close()