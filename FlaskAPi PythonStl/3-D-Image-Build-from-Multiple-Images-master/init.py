
import copy
import json
import os
import shutil
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import Image
import import_ipynb
from rotate_translate import node
from parent_child_tree import createtree
from multipleoperation import *
from subprocess import run
import open3d as o3d
import time
from pyntcloud import PyntCloud
from flask import Flask, request, jsonify
from flask_cors import CORS
import aiofiles
import aiofiles.os
import urllib.request


app = Flask(__name__)
CORS(app)



def detect(c):
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * perimeter , True)
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        if (ar >= 0.999 and ar <= 1.001):
            shape = "square" 
        else:
            shape = "rectangle"
    elif len(approx) == 5:
        shape = "pentagon"
    elif len(approx) == 6:
        shape = "hexagon"
    else:
        shape = "circle"
    return shape



async def Dimensioning(userId, view, image):
    await aiofiles.os.makedirs('static/temp', exist_ok=True)
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    ratio = 0
    shape = "unidentified"
    try:
        w, h, _ = img.shape
        drawSize = int(h / 300)
        imgrey = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgrey, 127, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy, contours = zip(*sorted(zip(hierarchy[0], contours), key=lambda x: cv2.contourArea(x[1]), reverse=True))
        for i, c in enumerate(contours):
            if hierarchy[i][3] != -1 or (hierarchy[i][3] == -1 and hierarchy[i][2] == -1):
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    x, y, w, h = cv2.boundingRect(c)
                    print(detect(c))
                    shape = detect(c)
                    print(shape)
                    if shape == "unidentified":
                        continue
                    if shape == "triangle" or shape == "pentagon" or shape == "hexagon":
                        img = cv2.drawContours(img, [box], 0, (0, 0, 255), drawSize)
                    if shape == "circle":
                        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), drawSize)
                        cv2.line(img, (x, y), (x + w, y), (0, 255, 0), 2)
                        ratio = 1 / w
                    else:
                        cv2.line(img, tuple(box[0]), tuple(box[1]), (0, 255, 0), 2)
                        ratio = 1.0 / rect[1][1]
                    break
        folder = 'static/temp/' + userId
        await aiofiles.os.makedirs(folder, exist_ok=True)
        try:
            await aiofiles.os.remove(folder + '/' + view + '.jpg')
        except:
            pass
        path_file = (folder + '/' + view + '.jpg')
        small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        async with aiofiles.open(path_file, "wb") as f:
            await f.write(cv2.imencode('.jpg', small)[1])
        ratio = str(ratio)
        data = {'image': path_file, 'shape': shape, 'ratio': ratio}
        return data
    except Exception as e:
        print(e)
        


async def Convert(userId, front_image, side_image, top_image, fratio, sratio, tratio):
    start = time.time() 
    await aiofiles.os.makedirs('static/'+userId, exist_ok=True)
    img_front = cv2.imread(front_image,cv2.IMREAD_UNCHANGED)
    img_side = cv2.imread(side_image,cv2.IMREAD_UNCHANGED)
    img_top = cv2.imread(top_image,cv2.IMREAD_UNCHANGED)
    fratio = float(fratio)
    sratio = float(sratio)
    tratio = float(tratio)
    primitive = []
    object_front = valid_contours(img_front,"front",fratio)
    re_arrange(object_front,"front")
    object_side = valid_contours(img_side,"side",sratio)
    re_arrange(object_side,"side")
    object_top = valid_contours(img_top,"top",tratio)
    re_arrange(object_top,"top")
    minApprox = 0.05
    primitive = combining(object_front,object_side,object_top,minApprox)
    final = []
    for set in primitive:
        for shape in set:
            final.append(shape[0])
    
    try:
        await aiofiles.os.remove('static/' + userId + "/" + userId + '.scad')
    except: pass
    path_file = ('static/' + userId + "/" + userId + '.scad')
    if(len(final) == 0):
        path_file = 'static/error.txt'
        # f = await aiofiles.open(path_file, mode='a')
        # f.write("Cannot determine the 3d geometry, check your files again!")
        # f.close()
        async with aiofiles.open(path_file, mode='a') as f:
            await f.write("Cannot determine the 3d geometry, check your files again!")
    
    createtree(final,path_file)
    end = time.time() 
    print("Total time taken to convert:",end-start)
    return path_file


async def remove_static_directory(directory):
    try:
        shutil.rmtree(directory)
        print("Directory 'static' removed successfully.")
    except Exception as e:
        print(f"Error removing directory: {e}")



def download_image(url, save_path):
    urllib.request.urlretrieve(url, save_path)
    
async def delete_files(file):
    try:
        await aiofiles.os.remove(file)
        print(f"Deleted {file}")
    except FileNotFoundError:
        print(f"File {file} not found")


userId="1"
front_image = "TestBench/"+userId+"/front.jpg"
side_image = "TestBench/"+userId+"/side.jpg"
top_image = "TestBench/"+userId+"/top.jpg"




async def formation(frontView, sideView, topView):
# async def formation():
    await remove_static_directory('static')
    await delete_files(front_image)
    await delete_files(side_image)
    await delete_files(top_image)

    
    download_image(frontView, front_image)
    download_image(sideView, side_image)
    download_image(topView, top_image)
    
    
    d1 = await Dimensioning(userId,"front",front_image)
    d2 = await Dimensioning(userId,"side",side_image)
    d3 = await Dimensioning(userId,"top",top_image)

    fratio = float(d1["ratio"]) * 2 
    sratio = float(d2["ratio"]) * 2
    tratio = float(d3["ratio"]) * 2

    file_name = await Convert(userId,front_image, side_image, top_image,fratio, sratio, tratio)
    return file_name
    

    
    

def stlFileGeneration():

    print("please wait generating 3D view...") 
    start = time.time() 
    scad_filename = "static/" + userId + '/' + userId + ".scad"
    stl_filename = "static/" + userId + '/' + userId + ".stl"
    pcd_filename = "static/" + userId + '/' + userId + ".pcd"
    if(os.path.isfile(scad_filename)):
        run("\"C:/Program Files/OpenSCAD/openscad.exe\" -o " + stl_filename + " " + scad_filename)
        if(os.path.isfile(stl_filename)):
            print("stl file generated at " + stl_filename)
            return stl_filename
        else:
            print("file not found")
    else:
        print("scad file not found")
        
        
        
        
    
@app.route('/getModel', methods=['GET'])
async def get_model_answer():
    # Get the user input from the query parameters
    user_input = request.args.get('query')
    
    # Get the URLs from the query parameters
    frontView = request.args.get('front')
    sideView = request.args.get('side')
    topView = request.args.get('top')

    file_name = await formation(frontView, sideView, topView)
    # file_name = await formation()

    stl_file = stlFileGeneration()
    
    # Return the response as JSON
    return jsonify({'response': "Model is ready "+ file_name+" Stl file as "+ stl_file})







if __name__ == '__main__':
    app.run(host='192.168.29.190')