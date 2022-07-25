import numpy as np
from flask import Flask, request,render_template, jsonify
from tensorflow import keras
import numpy
import cv2 as ocv
model = keras.models.load_model('my_model.h5')
classes = ['NORMAL','PNEUMONIA']

def predict_label ( image_path ):
    x = ((ocv.resize(ocv.imread(image_path, 0), (100, 100)) / 255))
    print(np.shape(x))
    x = x.reshape(1, 100, 100, 1)
    res = model.predict_on_batch(x)
    classification = numpy.where(res == numpy.amax(res))[1][0]
    return classification


app = Flask(__name__)

@app.route('/')
def indix():
    return render_template('revision.html')


@app.route('/submit',methods=['GET','POST'])
def revision():
    if request.method == 'POST':
        img = request.files['my_image']
        import os
        image_path = "static/"+img.filename
        os.chmod("static/",755)
        img.save(image_path)
        p = predict_label(image_path)
        pp=classes[p]
        print(image_path)
    return render_template('revision2.html',prediction=pp,image_path=image_path)
if __name__=="__main__":
    app.run(debug=True)