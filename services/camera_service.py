import cv2
import numpy as np

camera = cv2.VideoCapture(0)
camera_on = True


def toggle_camera():
    global camera_on
    camera_on = not camera_on
    return camera_on


def get_camera_status():
    return camera_on


def generate_frames():
    global camera_on, camera

    while True:
        if camera_on:
            success, frame = camera.read()
            if not success:
                break
        else:
            frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')