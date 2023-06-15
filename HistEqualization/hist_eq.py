import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv2.imread("moon.bmp")
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (ht, width) = I.shape
    I_array = I.flatten()
    #Get the histogram of the flattened image array along with its correspoding bin(0-255)
    hist, bins = np.histogram(I_array, 255)
    #calculate the cumulative histogram for the above
    cumu = np.cumsum(hist)
    #normalizing the cumulative sum from 0-255
    n_cum = cumu*255/cumu.max()
    L = np.zeros((ht,width))
    for i in range(ht):
        for j in range(width):
            #find the index at which this pixel value appears from 0-255
            it = np.where(bins == I[i,j])[0][0] -1
            #assign the corresponding value to the resulting image
            L[i,j]=n_cum[it]
    res=L.astype(np.uint8)
    moon = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img)
    axarr[1].imshow(moon)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__': 
    main()