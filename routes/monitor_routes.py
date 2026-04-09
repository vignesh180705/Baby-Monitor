from flask import Blueprint, render_template, Response, jsonify
from services.camera_service import generate_frames, toggle_camera, get_camera_status
from services.audio_service import (
    toggle_audio,
    get_audio_status,
    start_monitoring,
    stop_monitoring,
    is_monitoring,
    generate_audio_stream
)

monitor = Blueprint('monitor', __name__)


@monitor.route('/baby-monitor')
def baby_monitor():
    return render_template('baby_monitor.html')


@monitor.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@monitor.route('/toggle_camera', methods=['POST'])
def toggle_camera_route():
    status = toggle_camera()
    return jsonify({"message": f"Camera {'ON' if status else 'OFF'}"})


@monitor.route('/toggle_audio', methods=['POST'])
def toggle_audio_route():
    status = toggle_audio()
    return jsonify({"message": f"Microphone {'ON' if status else 'MUTED'}"})


@monitor.route('/start_monitoring', methods=['POST'])
def start_monitoring_route():
    started = start_monitoring()
    if started:
        return jsonify({"message": "Monitoring started."})
    return jsonify({"message": "Monitoring is already running."})


@monitor.route('/stop_monitoring', methods=['POST'])
def stop_monitoring_route():
    stop_monitoring()
    return jsonify({"message": "Monitoring stopped."})

@monitor.route('/audio_feed')
def audio_feed():
    return Response(generate_audio_stream(),
                    mimetype='audio/wav')