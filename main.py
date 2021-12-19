from flask import Flask, render_template, request
import util
import uuid
import mahotas as mt
import os
import cv2
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
leaf_category_names = ['pubescent bamboo','Chinese horse chestnut','Anhui Barberry', 'Chinese redbud','true indigo','Japanese maple','Nanmu',' castor aralia', 'Chinese cinnamon','goldenrain tree','Big-fruited Holly','Japanese cheesewood', 'wintersweet','camphortree','Japan Arrowwood','sweet osmanthus','deodar','ginkgo, maidenhair tree', 'Crape myrtle, Crepe myrtle','oleander','yew plum pine','Japanese Flowering Cherry','Glossy Privet','Chinese Toon','peach','Ford Woodlotus','trident maple','Beales barberry','southern magnolia','Canadian poplar','Chinese tulip tree','tangerine']

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html',msg="", img1='', img2='', img3='', img4='')

attributes = ['area','perimeter','length','width','aspect_ratio','rectangularity','circularity','red_mean','green_mean','blue_mean','stddev_red','stddev_green','stddev_blue','contrast','correlation','inverse_difference_moments','entropy']
def find_result(f_name):
    imgpath = './static/Uploads/' + f_name
    main_img = cv2.imread(imgpath)
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (25, 25), 0)
    ret_otsu, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((50, 50), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0
    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)
    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    Mom = cv2.moments(cnt)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    rectangularity = w * h / area
    circularity = ((perimeter) ** 2) / area
    textures = mt.features.haralick(gray)
    ht_mean = textures.mean(axis=0)
    contrast = ht_mean[1]
    correlation = ht_mean[2]
    inverse_diff_moments = ht_mean[4]
    entropy = ht_mean[8]
    vector=[area, perimeter, w, h, aspect_ratio, rectangularity, circularity, red_mean, green_mean,blue_mean, red_std, green_std, blue_std, contrast, correlation, inverse_diff_moments,entropy]
    return vector

@app.route("/test_image", methods=['POST'])
def test_image():
    msg=''
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']
        file3 = request.files['file3']
        file4 = request.files['file4']
        filename1 = secure_filename(file1.filename)
        extension1 = os.path.splitext(filename1)
        filename2 = secure_filename(file1.filename)
        extension2 = os.path.splitext(filename2)
        filename3 = secure_filename(file1.filename)
        extension3 = os.path.splitext(filename3)
        filename4 = secure_filename(file1.filename)
        extension4 = os.path.splitext(filename4)
        allowed_extensions = {'.jpg'}
        if extension1[1] in allowed_extensions and extension2[1] in allowed_extensions and extension3[1] in allowed_extensions and extension4[1] in allowed_extensions:
            f_name1 = str(uuid.uuid4()) + str(extension1[1])
            app.config['UPLOAD_FOLDER'] = 'static/Uploads'
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name1))
            f_name2 = str(uuid.uuid4()) + str(extension2[1])
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name2))
            f_name3 = str(uuid.uuid4()) + str(extension3[1])
            file3.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name3))
            f_name4 = str(uuid.uuid4()) + str(extension4[1])
            file4.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name4))
            df = pd.DataFrame([], columns=attributes)
            img_names_list=[f_name1,f_name2,f_name3,f_name4]
            for i in img_names_list:
                vector=find_result(i)
                temp = pd.DataFrame([vector], columns=attributes)
                df=df.append(temp)
            result=util.get_leaf_name(df)
            img1 = leaf_category_names[result[0]]
            img2 = leaf_category_names[result[1]]
            img3 = leaf_category_names[result[2]]
            img4 = leaf_category_names[result[3]]
            msg="Successfully classified! Results for each is shown in their respective colors."
            print(result)
            return render_template('index.html',msg=msg,img1=img1,img2=img2,img3=img3,img4=img4)
        else:
            msg="Upload only .jpg file!"
            return render_template('index.html', msg=msg, img1='', img2='', img3='', img4='')

if __name__ == "__main__":
    util.load_model()
    app.run(debug=True)