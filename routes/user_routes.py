from flask import Flask, request, Blueprint, render_template
import subprocess
import os
app = Flask(__name__)

user = Blueprint('user', __name__)

@user.route('/user')
def user_page():
    print("User page requested")
    return render_template('user.html')


@user.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    file.save(f'uploads/{file.filename}')
    return f'Saved as uploads/{file.filename}'


'''@user.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    
    webm_path = os.path.join('uploads', file.filename)
    mp3_filename = file.filename.replace('.webm', '.mp3')
    mp3_path = os.path.join('uploads', mp3_filename)

    file.save(webm_path)

    subprocess.run([
        'ffmpeg', '-i', webm_path,
        '-vn', '-ab', '192k', '-ar', '44100', '-y',
        mp3_path
    ])

    return f'Saved as {mp3_path}'
'''

if __name__ == "__main__":
    import os
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
