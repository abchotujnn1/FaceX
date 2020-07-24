import os
import cv2
import base64
import numpy as np
from face_recognition import face_encodings,load_image_file,face_locations,face_landmarks,face_recognition_cli
from PIL import Image
import cvlib as cv
from flask import Flask,render_template,request,jsonify,redirect,url_for,send_from_directory
from werkzeug.utils import  secure_filename
BASE_DIR=os.path.dirname(os.path.abspath(__name__))


app=Flask(__name__)

UPLOAD_FOLDER='./uploads'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER


def image_to_vector(img_path):
    pixel=load_image_file(img_path)
    encoding=face_encodings(pixel)[0]
    return encoding

def base64_to_pixel(base64_string):
    img_base64_string=base64_string.split(b',')[1]
    # img_base64_bytes=bytes(img_base64_string,'utf-8')
    img_base64_bytes=img_base64_string
    img_decode=base64.b64decode(img_base64_bytes)
    img_decode_array=np.asarray(bytearray(img_decode),dtype='uint8')
    img_pixel=cv2.imdecode(img_decode_array,1)
    return img_pixel

def coordinate(img):
        img=load_image_file(img)
        coordinate=face_locations(img)
        return coordinate

def landmark(img):
    img = load_image_file(img)
    location = face_landmarks(img)
    return location

def gender_detect(img):
    image=cv2.imread(img)
    face, conf = cv.detect_face(image)
    padding = 20
    for f in face:
        (startX, startY) = max(0, f[0] - padding), max(0, f[1] - padding)
        (endX, endY) = min(image.shape[1] - 1, f[2] + padding), min(image.shape[0] - 1, f[3] + padding)
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        face_crop = np.copy(image[startY:endY, startX:endX])
        (label, confidence) = cv.detect_gender(face_crop)
        idx = np.argmax(confidence)
        conf=np.max(confidence).item()
        label = label[idx]
        return {"conf":conf,"label":label}


@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')


#face_encoding
@app.route('/encoding', methods=['GET','POST'])
def face_encoding():
    if request.method=="POST":
        f=request.files['file']
        filename=secure_filename(f.filename)
        print(filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        img_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
        encoding=image_to_vector(img_path)
        return jsonify({"encoding":encoding.tolist()})
    return render_template('about.html')


#coordinate of faces
@app.route('/face_detection', methods=['GET','POST'])
def face_detection():
    if request.method=="POST":
        f=request.files['file']
        filename=secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        img_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
        coord=coordinate(img_path)
        return jsonify({'coord':coord})
    return render_template('about.html')

#face_landmark
@app.route('/face_landmark',methods=['GET','POST'])
def face_landmark():
    if request.method=="POST":
        f=request.files['file']
        filename=secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        img_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
        landM=landmark(img_path)
        return jsonify({"landM":landM})
    return render_template('about.html')

@app.route('/gender',methods=["GET","POST"])
def gender():
    if request.method=="POST":
        f=request.files['file']
        filename=secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        img_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
        gen=gender_detect(img_path)
        return jsonify({"gen":gen})
    return render_template('about.html')

@app.route('/age')
def age():
    pass

@app.route('/face_match')
def face_match():
    pass

@app.route('/face_recognition')
def face_recognition():
    pass

@app.route('/spoof_detection')
def spoof_detection():
    pass

@app.route('/face_tracking')
def face_tracking():
    pass

if __name__=="__main__":
    app.run(debug=True)
