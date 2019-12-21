import os
import cv2
import numpy as np
I=np.zeros([1, 2500])
for root, dirs, files in os.walk(r'C:\Users\priya\OneDrive\Documents\CV\resized_img_comdata'): # root: root folder, dirs: sub-folder, files: image files inside sub-folder
    for file in files:
        F=os.path.join(root,file) # Path from root directory to image file
        img = cv2.imread(str(F)) # Read the image in RGB format
        if np.shape(np.shape(img))[0] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image from RGB to Grayscale
        img = np.resize(img, (50, 50))  # Resize the grayscale image
        I1 = np.reshape(img, (1, np.size(img))) # Convert an Image matrix to a vector
        I = np.concatenate((I, I1), axis=0) # Concatenate the vector obtained from the previous step to a new matrix I
I = I[1:, 0:]
print(I) # This is the final matrix which will be used as an input for Random Projection. Random Projection is a method to reduce features by reducing the dimension.
print(np.shape(I))

np.savetxt("ImageProcessingComData.csv", I, delimiter=",", fmt='%s') # Save the matrix in csv format