# Import required libraries.
import os
from flask import request, render_template, Flask, jsonify
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import requests
import json
from PIL import Image
from io import BytesIO

# Generate timestamp.
timestr = time.strftime("%Y%m%d-%H%M%S")

# App configrations.
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xecsd]/'
app.config["UPLOAD_FOLDER"] = "static"

# Result dictionary to be sent to HTML.
results = {"PATH": 0, "MSG": ""}

# Prediction function.
def AnalyseAndDraw(img_url, img_path):
    subscription_key = "128ba37bdaa043118ade8a452875c59e"
    analyze_url = "https://imagedetection16.cognitiveservices.azure.com//vision/v3.1/analyze"
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    params = {'visualFeatures': 'Categories,Description,Faces,Objects'}
    data = {'url': img_url}

    try:
        # Get image analysis.
        response = requests.post(analyze_url, headers=headers,params=params, json=data)
        response.raise_for_status()
        analysis = response.json()

    except Exception as e:
        # Error handling.
        results = {"PATH": 0, "MSG": "FAIL"}
        return results
    
    # Get coordinates of faces in the image.
    faces = []
    for rec in analysis['faces']:
        k = []
        k.append(rec['faceRectangle']['left'])
        k.append(rec['faceRectangle']['top'])
        k.append(rec['faceRectangle']['width'])
        k.append(rec['faceRectangle']['height'])
        faces.append(k)
    
    # Draw rectangles around the faces detected in the image.
    image = Image.open(BytesIO(requests.get(img_url).content))
    np_img = np.asarray(image)
    drawing = np_img
    for i in range(len(faces)):
        color = (255,0,0)
        cv2.rectangle(drawing, (int(faces[i][0]), int(faces[i][1])), \
          (int(faces[i][0]+faces[i][2]), int(faces[i][1]+faces[i][3])), color, 2)

    # Save the image.
    plt.imsave(img_path, drawing)

    # Return success message and image path.
    results = {"PATH": img_path, "MSG": "SUCCESS"}
    return results



# Define functions to be executed at endpoints.
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            url = request.form['url']  # Get the URL.
            filename = timestr + '.png' # Make a file name from time string.
            path = os.path.join("static", filename)
            result = AnalyseAndDraw(img_url=url, img_path=path)  # Send the image to prediction algorithm.
            return render_template("index.html", res=result)
        except:
            return render_template("index.html", res=result)

    return render_template("index.html", res=results)


# Run app.
if __name__ == "__main__":
    app.run(debug=True)  # Set debug = False in production.
