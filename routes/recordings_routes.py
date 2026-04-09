from flask import Blueprint, render_template, send_from_directory
import os

from services.baby_service import get_cry_result
from models.audio_model import get_audio_result, insert_audio_result

recordings = Blueprint('recordings', __name__)

UPLOAD_FOLDER = "uploads"


@recordings.route('/recordings')
def view_recordings():
    files_data = []

    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):

            if filename.endswith((".wav", ".webm", ".mp3")):
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                db_result = get_audio_result(filename)

                if db_result:
                    status = db_result["predicted_label"]
                    confidence = {
                        "Non_Cry": db_result["non_cry"],
                        "Cry": db_result["cry"]
                    }

                else:
                    result = get_cry_result(filepath)

                    status = result["predicted_label"]
                    confidence = result["confidence"]

                    insert_audio_result(
                        filename,
                        status,
                        confidence["Non_Cry"],
                        confidence["Cry"]
                    )

                files_data.append({
                    "filename": filename,
                    "status": status,
                    "confidence": confidence
                })

    files_data.sort(key=lambda x: x["filename"], reverse=True)

    return render_template('recordings.html', files=files_data)
@recordings.route('/download/<filename>')
def download_recording(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)