from dotenv import load_dotenv

load_dotenv()

import os
import random

import cv2
import imutils
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from tqdm import tqdm

## NOTES: Run only when you have successfully added 10 fake ticket images to the food_ticket_fake folder!
## Otherwise, you WILL encounter errors later!

def data_augmentation(dataset_path, augmented_path):

  #Iterate through each image we have 
  for imgName in tqdm(os.listdir(dataset_path)):
    #Below is basically the image opening process
    img_path = dataset_path + "/" + imgName
    img = plt.imread(img_path)
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]

    #This code below is specifically for brightness adjust (uses PIL)
    img_for_shading = Image.open(img_path)

    #Data augmentation loop
    #We augment each photo 10 times per effect
    for i in range(int(os.environ["N_AUGMENTATIONS"])):

      #--------------------------Horizontal Translation-------------------------#
      hori_translate = random.randint(-int(imgWidth / 10), int(imgWidth / 10))  #Generate random number to translate by
                                                                                #Divide by 10 to avoid major translations 
                                                                                #(i.e. large portions of the tickets go missing)
      #print(f"Translated horizontally by {hori_translate} pixels.")
      translated = imutils.translate(img, hori_translate, 0)                    #Do the translation

      #Name and save the translated image into the augmented folder
      cv2.imwrite(augmented_path + "/" + imgName.split("_")[1].split(".")[0] + 
                  "_horiTranslate_" + str(i) + ".jpg", cv2.cvtColor(translated, cv2.COLOR_RGB2BGR))

      #--------------------------Vertical Translation-------------------------#
      verti_translate = random.randint(-int(imgHeight / 10), int(imgHeight / 10)) #Generate random number to translate by
                                                                                  #Divide by 10 to avoid major translations 
                                                                                  #(i.e. large portions of the tickets go missing)
      #print(f"Translated vertically by {verti_translate} pixels.")
      translated = imutils.translate(img, 0, verti_translate)                    #Do the translation

      #Name and save the translated image into the augmented folder
      cv2.imwrite(augmented_path + "/" + imgName.split("_")[1].split(".")[0] + 
                  "_vertiTranslate_" + str(i) + ".jpg", cv2.cvtColor(translated, cv2.COLOR_RGB2BGR))    

      #--------------------------Rotation---------------------------------------#

      #Randomly rotate the image 
      rotation = random.randint(0, 360)   #Generate a random rotation angle
      #print(f"Rotated by {rotation} degrees.")
      rotated = imutils.rotate(img, angle = rotation)

      #Save the rotated image into the augmented folder 
      cv2.imwrite(augmented_path + "/" + imgName.split("_")[1].split(".")[0] + "_rotate_" + str(i) + ".jpg", cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))

      #--------------------------Brightness Adjust------------------------------#

      #Randomly discolour the image
      brightness = random.uniform(0.5,1.5) #Generate a random shade

      #image brightness enhancer
      enhancer = ImageEnhance.Brightness(img_for_shading)
      im_output = enhancer.enhance(brightness)

      #Save the rotated image into the augmented folder 
      im_output.save(augmented_path + "/" + imgName.split("_")[1].split(".")[0] + "_shaded_" + str(i) + ".jpg")

if __name__ == "__main__":
	data_augmentation(os.environ["REAL_FOOD_TICKET_PATH_DATASET"], os.environ["REAL_FOOD_TICKET_PATH_PREPROCESSED"])
	#data_augmentation("/content/drive/Shareddrives/GEC2023HACKATHON/Students/Team7/dataset/food_ticket_real")
	#data_augmentation("/content/drive/Shared drives/GEC2023HACKATHON/Students/Team##/dataset/food_ticket_fake")