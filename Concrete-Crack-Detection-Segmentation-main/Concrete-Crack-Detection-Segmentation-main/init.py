import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms
from torchvision import transforms
import os


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




def predict_on_crops(input_image, height=227, width=227, save_crops = False):
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
                if not os.path.exists(os.path.join('real_images', folder_name)):
                    os.makedirs(os.path.join('real_images', folder_name))
                filename = os.path.join('real_images', folder_name,'img_{}.png'.format(k))
                cv2.imwrite(filename, add_img)
            output_image[i:i+height, j:j+width,:] = add_img
            k+=1
    ## Save output image
    cv2.imwrite(os.path.join('real_images','predictions', folder_name+ '.jpg'), output_image)
    return output_image



plt.figure(figsize=(10,10))
output_image = predict_on_crops('real_images/concrete_crack7.jpg', 128, 128)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))