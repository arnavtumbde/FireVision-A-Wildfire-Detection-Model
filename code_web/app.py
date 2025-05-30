import os
import cv2
import math
import numpy as np
import base64
from threading import Thread
from flask_socketio import SocketIO, emit
from flask import Flask, render_template, request, jsonify
from fire_flow import fire_pixel_segmentation, fire_flow
from smoke_flow import smoke_pixel_segmentation, smoke_flow
from yolo_detection import run_yolo, load_model
from analysis import graph

app = Flask(__name__)
socketio = SocketIO(app)

load_model()

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

threads = True
size_factor = 1
mFrames = 180
nFrames = 30

@socketio.on('stop_processing')
def stop_stream():
    global threads
    threads = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('index.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    global threads
    threads = True
    
    video_file = request.files['video']
    #changes
    video_url = request.form.get('videoUrl')
    video_path = None  # ✅ Initialize

    # ✅ If user entered URL
    if video_url:
        video_path = video_url  # URL assigned directly
        print(f"Processing video from URL: {video_path}")
    
    elif video_file and video_file.filename != '':

        filename = video_file.filename
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)

    global size_factor, mFrames, nFrames
    size_factor = int(request.form.get('sizeFactor'))
    mFrames = int(request.form.get('mFrames'))
    nFrames = int(request.form.get('nFrames'))
    
    is_camera_stable = request.form.get('cameraStable') == 'on'
    run_yolo_detection = request.form.get('runYolo') == 'on'
    
    if is_camera_stable and run_yolo_detection:
        thread1 = Thread(target=process_stable_camera, args=(video_path,))
        thread2 = Thread(target=process_yolo_detection, args=(video_path,))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        if video_file and video_file.filename:
            os.unlink(video_path)

        return jsonify({'result': "Video successfully processed"})
    elif is_camera_stable:
        process_stable_camera(video_path)
        os.unlink(video_path)

        return jsonify({'result': "Video successfully processed"})
    elif run_yolo_detection:
        process_yolo_detection(video_path)
        os.unlink(video_path)

        return jsonify({'result': "Video successfully processed"})
    else:
        os.unlink(video_path)

        return jsonify({'error': 'Error processing video'})
    

def process_stable_camera(video_path):
    global size_factor

    cap = cv2.VideoCapture(video_path)
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))

    _, first_frame = cap.read()
    height, width = first_frame.shape[:2]
    new_width = width // size_factor
    new_height = height // size_factor
    first_frame = cv2.resize(first_frame, (new_width, new_height))

    # Fire Variables
    areas = [0]
    area_frame = np.zeros_like(first_frame)
    fireX = []
    fireY = []
    path = []

    # Smoke Variables
    prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    wind_dir = []
    smoke_dir = []

    while cap.isOpened() and threads:
        ret, frame = cap.read()
        if not ret:
            break
        im = cv2.resize(frame, (new_width, new_height))
        processed_frame = im.copy()
        
        #######################################
        fire_mask = fire_pixel_segmentation(im)
        area, new_area_frame, processed_frame, end_point = fire_flow(fire_mask, area_frame, fireX, fireY, processed_frame, mFrames)
        area_frame = new_area_frame
        
        if end_point != None:
            path.append(end_point)
        
        areas.append(area)
        
        #########################################
        smoke_mask = smoke_pixel_segmentation(im)
        curr, processed_frame, smoke = smoke_flow(im, prev, smoke_mask, wind_dir, processed_frame, nFrames)
        prev = curr
        
        if smoke != None:
            smoke_dir.append(smoke)
        
        if len(path) > 2:
            start_point = path[0]
            end_point = path[-1]
            cv2.arrowedLine(processed_frame, start_point, end_point, (0, 255, 0), 2, tipLength=0.5)

            delta_x = end_point[0] - start_point[0]
            delta_y = end_point[1] - start_point[1]

            angle_rad = np.arctan2(delta_y, delta_x)
            angle_deg = np.degrees(angle_rad)

            text_position = (start_point[0] + 10, start_point[1] - 10)
            cv2.putText(processed_frame, f'Fire Spread Direction: {angle_deg:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # cv2.imshow('Frame', im)
        
        # cv2.imshow('Fire mask', fire_mask)
        # cv2.imshow('Smoke mask', smoke_mask)
        # cv2.imshow('Fire flow', processed_frame)

        f_mask = cv2.cvtColor(fire_mask, cv2.COLOR_GRAY2BGR)
        s_mask = cv2.cvtColor(smoke_mask, cv2.COLOR_GRAY2BGR)

        row1 = np.hstack((im, f_mask))  
        row2 = np.hstack((s_mask, processed_frame))

        final_frame = np.vstack((row1, row2))
        # print(final_frame.shape)

        _, buffer = cv2.imencode('.jpg', final_frame)
        img_str = base64.b64encode(buffer).decode('utf-8')

        socketio.emit('stable_update', img_str)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()

    if(len(areas) > 1):
        points = []
        seen = set()

        for item in path:
            if item not in seen:
                points.append(item)
                seen.add(item)

        frame_numbers = np.arange(len(areas))  
        time_seconds = frame_numbers / fps
        graph_str = graph(time_seconds, areas, smoke_dir, points)

        socketio.emit('analysis', graph_str)


def process_yolo_detection(video_path):

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened() and threads:
        ret, frame = cap.read()
        if not ret:
            break

        yolo_frame = run_yolo(frame)

        _, buffer = cv2.imencode('.jpg', yolo_frame)
        img_str = base64.b64encode(buffer).decode('utf-8')

        socketio.emit('yolo_update', img_str)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()

if __name__ == '__main__':
    socketio.run(app, debug=True)
