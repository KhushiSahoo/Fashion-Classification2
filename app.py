from flask import Flask,render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from flask_cors import CORS
import os
#.\env\Scripts\activate
app = Flask(__name__)
CORS(app)
dic = {0 : 'tshirt', 1 : 'trouser', 2:'pullover', 3:'dress', 4:'coat', 5:'sandal',6:'shirt', 7:'sneaker',8:'bag', 9:'boot'}
#0 => T-shirt/top 1 => Trouser 2 => Pullover 3 => Dress 4 => Coat 5 => Sandal 6 => Shirt 7 => Sneaker 8 => Bag 9 => Ankle boot
model = load_model('final_model29.h5')

def predict_label(img_path):
    img = load_img(img_path, color_mode="grayscale", target_size=(28, 28))
    # convert to array
    img=img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    result = model.predict(img)
    p=np.argmax(result[0])
    print(p)
    print(dic[p])
    os.remove(img_path)
    return dic[p]


 #routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/submit2", methods = ['GET', 'POST'])
def get_output2():
    if request.method == 'POST':
        img = request.files['my_image']
         
        img_path = "static/" + img.filename	
        img.save(img_path)

        p = predict_label(img_path)

    
    return render_template("index.html", prediction = p, img_path = img_path)

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        prediction=""
        files = request.files.getlist("file")
        for img in files:
            print(img.filename)
            img_path = "static/" + img.filename	
            img.save(img_path)
            p = predict_label(img_path)
            prediction=p
            
    return {'data':prediction}
    #return render_template("index.html", prediction = p, img_path = img_path)    


if __name__=="__main__":
    app.run(debug=True)    