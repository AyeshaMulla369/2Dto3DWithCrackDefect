from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import os
from fastapi.responses import FileResponse
import requests





mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]
idx_to_class = {0:'Negative', 1:'Positive'}


## Define data augmentation and transforms
chosen_transforms = {'train': transforms.Compose([
        transforms.RandomResizedCrop(size=227),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
]), 'val': transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
]),
}


# Load the saved model for immediate prediction
# Define the model architecture
resnet50_loaded = models.resnet50()
fc_inputs_loaded = resnet50_loaded.fc.in_features
resnet50_loaded.fc = nn.Sequential(
    nn.Linear(fc_inputs_loaded, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 2)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


# Load the saved state dictionary
resnet50_loaded.load_state_dict(torch.load('model.pth', map_location=device))
# Set the model to evaluation mode
resnet50_loaded.eval()

# Now you can use resnet50_loaded for immediate prediction


def predict(model, test_image, print_class = False):
     # it uses the model to predict on test_image...
    transform = chosen_transforms['val']
     
    test_image_tensor = transform(test_image)
    # if torch.cuda.is_available(): # checks if we have a gpu available
    #     print("cuda")
    #     test_image_tensor = test_image_tensor.view(1, 3, 227, 227).cuda()
    # else:
        # print("CPU")
    test_image_tensor = test_image_tensor.view(1, 3, 227, 227)
    
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        # this computes the output of the model
        out = model(test_image_tensor)
        # this computes the probability of each classes.
        ps = torch.exp(out)
        # we choose the top class. That is, the class with highest probability
        topk, topclass = ps.topk(1, dim=1)
        class_name = idx_to_class[topclass.cpu().numpy()[0][0]]
        if print_class:
            print("Output class :  ", class_name)
    return class_name




async def predict_on_crops(input_image, height=227, width=227, save_crops = False):
    im = cv2.imread(input_image)
    imgheight, imgwidth, channels = im.shape
    k=0
    output_image = np.zeros_like(im)
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            a = im[i:i+height, j:j+width]
            ## discard image cropss that are not full size
            # predicted_class = predict(base_model,Image.fromarray(a))
            
            predicted_class = predict(resnet50_loaded,Image.fromarray(a))
            ## save image
            file, ext = os.path.splitext(input_image)
            image_name = file.split('/')[-1]
            folder_name = 'out_' + image_name
            ## Put predicted class on the image
            if predicted_class == 'Positive':
                color = (0,0, 255)
            else:
                color = (0, 255, 0)
            cv2.putText(a, predicted_class, (50,50), cv2.FONT_HERSHEY_SIMPLEX , 0.7, color, 1, cv2.LINE_AA) 
            b = np.zeros_like(a, dtype=np.uint8)
            b[:] = color
            add_img = cv2.addWeighted(a, 0.9, b, 0.1, 0)
            ## Save crops
            if save_crops:
                if not await os.path.exists(os.path.join('real_images', folder_name)):
                    await os.makedirs(os.path.join('real_images', folder_name))
                filename = await os.path.join('real_images', folder_name,'img_{}.png'.format(k))
                cv2.imwrite(filename, add_img)
            output_image[i:i+height, j:j+width,:] = add_img
            k+=1
    ## Save output image
    cv2.imwrite(os.path.join('real_images','predictions', folder_name+ '.jpg'), output_image)
    return output_image





app = FastAPI()

def calculate_histogram(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])
    return histogram

def segment_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])
    darkest_intensity = np.argmax(histogram)
    _, binary_image = cv2.threshold(grayscale_image, darkest_intensity, 255, cv2.THRESH_BINARY)
    return binary_image

def process_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(grayscale_image, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_length = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, [contour_length], -1, (0, 255, 0), 2)
    (x, y, w, h) = cv2.boundingRect(contour_length)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (x, y, w, h)

def process_red_parts(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(grayscale_image, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_length = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, [contour_length], -1, (0, 255, 0), 2)
    mask = np.zeros_like(grayscale_image)
    cv2.drawContours(mask, [contour_length], -1, 255, cv2.FILLED)
    mask_inv = cv2.bitwise_not(mask)
    red_parts = cv2.bitwise_and(image, image, mask=mask_inv)
    red_parts[np.where((red_parts == [0,0,0]).all(axis=2))] = [0,0,255]
    non_red_contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in non_red_contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = max(w, h)
        cv2.rectangle(red_parts, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(red_parts, (x + w, y + h), (x + w + length, y + h), (255, 0, 0), 2)
        cv2.putText(red_parts, f'Length: {length}px', (x, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(red_parts, f'Width: {min(w, h)}px', (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return cv2.cvtColor(red_parts, cv2.COLOR_BGR2RGB)

# Define a folder to save the images
SAVE_FOLDER = "requestImages/"

# Ensure the folder exists
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

def download_image(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)

@app.post("/histogram/")
async def histogram_route(url: str):
    # Download the image
    image_filename = os.path.join(SAVE_FOLDER, "uploaded_imageHistogram.png")
    download_image(url, image_filename)
    
    image = cv2.imread(image_filename)
    histogram = calculate_histogram(image)
    plt.figure(figsize=(10, 5))
    plt.plot(histogram, color='black')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.title('Histogram of Depth Analysis')
    plt.axis('off')
    plt.savefig('histogram.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    return StreamingResponse(io.BytesIO(open("histogram.png", "rb").read()), media_type="image/png")



@app.post("/segmented_image/")
async def segmented_image_route(url: str):
    # Download the image
    image_filename = os.path.join(SAVE_FOLDER, "uploaded_imageSegmented.png")
    download_image(url, image_filename)
    
    image = cv2.imread(image_filename)
    binary_image = segment_image(image)
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')
    plt.title('Segmented Depth of Parts')
    plt.savefig('segmented_image.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    return StreamingResponse(io.BytesIO(open("segmented_image.png", "rb").read()), media_type="image/png")

@app.post("/processed_image/")
async def processed_image_route(url: str):
    # Download the image
    image_filename = os.path.join(SAVE_FOLDER, "uploaded_imageProcess.png")
    download_image(url, image_filename)
    
    image = cv2.imread(image_filename)
    processed_image, _ = process_image(image)
    plt.figure(figsize=(10, 10))
    plt.imshow(processed_image)
    plt.axis('off')
    plt.title('Processed Image')
    plt.savefig('processed_image.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    return StreamingResponse(io.BytesIO(open("processed_image.png", "rb").read()), media_type="image/png")

@app.post("/red_parts/")
async def red_parts_route(url: str):
    # Download the image
    image_filename = os.path.join(SAVE_FOLDER, "uploaded_imageRed.png")
    download_image(url, image_filename)
    
    image = cv2.imread(image_filename)
    red_parts_image = process_red_parts(image)
    plt.figure(figsize=(10, 10))
    plt.imshow(red_parts_image)
    plt.axis('off')
    plt.title('Processed Image with Red Parts')
    plt.savefig('red_parts_image.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    return StreamingResponse(io.BytesIO(open("red_parts_image.png", "rb").read()), media_type="image/png")





@app.post("/predict_on_crops/")
async def predict_on_crops_route(url: str):
    # Download the image
    image_filename = os.path.join(SAVE_FOLDER, "uploaded_imagePred.png")
    download_image(url, image_filename)
    
    
    # Perform prediction
    output_image = await predict_on_crops(image_filename, height=128, width=128)
    
    # Display the output image using Matplotlib
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis
    
    # Convert the output image to bytes
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='jpg')
    img_bytes.seek(0)
    
    # Return the image bytes as a StreamingResponse
    return StreamingResponse(img_bytes, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    
    # Specify the IP address and port to listen on
    # Replace '0.0.0.0' with the desired IP address
    # For example, to listen on localhost, use '127.0.0.1'
    uvicorn.run(app, host="192.168.29.190", port=8000)