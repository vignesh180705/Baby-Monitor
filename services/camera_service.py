import cv2
import numpy as np

camera = None
camera_on = False


def toggle_camera():
    global camera_on,camera
    if camera_on:
        if camera is not None:
            camera.release()
            camera = None
    else:
        camera = cv2.VideoCapture(0)
    camera_on = not camera_on
    return camera_on


def get_camera_status():
    return camera_on


def generate_frames():
    global camera_on, camera

    while True:
        if camera_on:
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(0)

            success, frame = camera.read()

            if not success:
                frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        else:
            # Camera OFF → blank frame
            frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')