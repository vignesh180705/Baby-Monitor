import sounddevice as sd
import soundfile as sf
import threading
import time
import os
from datetime import datetime

audio_on = False
monitoring = False
recording_thread = None
mic_mode = "idle"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def toggle_audio():
    global audio_on
    audio_on = not audio_on
    return audio_on


def get_audio_status():
    return audio_on


def is_monitoring():
    return monitoring


def record_audio_clip(duration=6, samplerate=44100):
    global audio_on, mic_mode
    if not audio_on:
        print("Microphone muted. Skipping recording.")
        return None
    if mic_mode != "monitor":
        print("Mic not in monitoring mode. Skipping recording.")
        return None
    print("Recording audio...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(UPLOAD_FOLDER, f"audio_{timestamp}.wav")
    sf.write(filename, audio, samplerate)
    print(f"Saved: {filename}")
    return filename


def monitor_audio():
    global monitoring
    while monitoring:
        record_audio_clip(duration=6)
        print("Waiting for next cycle...")
        time.sleep(54)


def start_monitoring():
    global monitoring, recording_thread, mic_mode
    if monitoring:
        return False
    mic_mode = "monitor"
    monitoring = True
    recording_thread = threading.Thread(target=monitor_audio, daemon=True)
    recording_thread.start()
    return True


def stop_monitoring():
    global monitoring, mic_mode
    monitoring = False
    if mic_mode == "monitor":
        mic_mode = "idle"

def generate_audio_stream(samplerate=44100, blocksize=1024):
    global mic_mode

    if mic_mode == "monitor":
        print("Monitoring active. Audio stream blocked.")
        return

    mic_mode = "stream"

    import queue, struct
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype='int16',
        blocksize=blocksize,
        callback=callback
    )

    stream.start()

    def wav_header():
        datasize = 2000000000
        o = bytes("RIFF", 'ascii')
        o += struct.pack('<I', datasize + 36)
        o += bytes("WAVEfmt ", 'ascii')
        o += struct.pack('<IHHIIHH', 16, 1, 1, samplerate, samplerate * 2, 2, 16)
        o += bytes("data", 'ascii')
        o += struct.pack('<I', datasize)
        return o

    yield wav_header()

    try:
        while mic_mode == "stream" and audio_on:
            data = q.get()
            yield data.tobytes()
    except GeneratorExit:
        pass
    finally:
        stream.stop()
        stream.close()
        mic_mode = "idle"
