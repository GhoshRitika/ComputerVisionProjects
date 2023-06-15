import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def train_hsv(path):
    images = os.listdir(path)
    H = []
    S = []
    for img_name in images:
        img_path=os.path.join(path, img_name)
        img = cv2.imread(img_path)
        I = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                if(I[i][j][1]>35):
                    H.append(I[i][j][0])
                    S.append(I[i][j][1])
    hist, x, y = np.histogram2d(H, S, bins=[np.arange(257),np.arange(257)])
    norm = hist/hist.max()
    plt.imshow(norm)
    plt.colorbar()
    plt.show()
    np.save("histogram.npy", norm)
    return norm

def train_rgb(path):
    images = os.listdir(path)
    R = []
    G = []
    B = []
    for img_name in images:
        img_path=os.path.join(path, img_name)
        img = cv2.imread(img_path)
        I = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                if(I[i][j][0]< 215  or I[i][j][1]< 215 or I[i][j][2]< 215):
                    # if(I[i][j][0]<150):
                    R.append(I[i][j][0])
                    G.append(I[i][j][1])
                    B.append(I[i][j][2])
    hist_RG, x, y = np.histogram2d(R, G, bins=[np.arange(257),np.arange(257)])
    norm_RG = hist_RG/hist_RG.max()
    plt.imshow(norm_RG)
    plt.colorbar()
    plt.show()
    np.save("histogram_rg.npy", norm_RG)
    hist_GB, x, y = np.histogram2d(G, B, bins=[np.arange(257),np.arange(257)])
    norm_GB = hist_RG/hist_GB.max()
    plt.imshow(norm_GB)
    plt.colorbar()
    plt.show()
    np.save("histogram_gb.npy", norm_GB)
    return norm_RG, norm_GB

def skin_segmentation(img, hist, thresh):
    I = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (ht, w, d) = I.shape
    L = np.zeros((ht,w,d))
    for i in range(ht):
        for j in range(w):
            hue=I[i][j][0]
            sat=I[i][j][1]
            #histogram works like a look up table here
            if(hist[hue][sat]>thresh):
                L[i][j][0]=img[i][j][0]
                L[i][j][1]=img[i][j][1]
                L[i][j][2]=img[i][j][2]
            else:
                L[i][j][0]=0
                L[i][j][1]=0
                L[i][j][2]=0
    return L

def skin_segmentation(I, hist1, hist2, thresh1, thresh2):
    # I = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (ht, w, d) = I.shape
    L = np.zeros((ht,w,d))
    for i in range(ht):
        for j in range(w):
            red=I[i][j][0]
            green=I[i][j][1]
            blue=I[i][j][2]
            #histogram works like a look up table here
            if(hist1[red][green]>thresh1 or hist2[green][blue]>thresh2):
                L[i][j][0]=I[i][j][0]
                L[i][j][1]=I[i][j][1]
                L[i][j][2]=I[i][j][2]
            else:
                L[i][j][0]=0
                L[i][j][1]=0
                L[i][j][2]=0
    return L

def main():
    path = '/home/ritz/SpringQ/CV/MP4/Hands'
    # hist_hsv=train_hsv(path)
    hist_hsv=np.load("histogram.npy")
    # hist_RG, hist_GB = train_rgb(path)
    # hist_RG=np.load("histogram_rg.npy")
    # hist_GB=np.load("histogram_gb.npy")
    thresh_HSV=0.028 
    # thresh_RG=0.025
    # thresh_GB=0.025
    img1 = cv2.imread("joy1.bmp")
    rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    L1 = skin_segmentation(img1, hist_hsv, thresh_HSV)
    res1=L1.astype(np.uint8)
    result1 = cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread("gun1.bmp")
    rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    L2 = skin_segmentation(img2, hist_hsv, thresh_HSV)
    res2=L2.astype(np.uint8)
    result2 = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)
    img3 = cv2.imread("pointer1.bmp")
    rgb3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    L3 = skin_segmentation(img3, hist_hsv, thresh_HSV)
    res3=L3.astype(np.uint8)
    result3 = cv2.cvtColor(res3, cv2.COLOR_BGR2RGB)
    f, axarr = plt.subplots(3,2)
    # img1 = cv2.imread("joy1.bmp")
    # rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # res1 = skin_segmentation(rgb1, hist_RG, hist_GB, thresh_RG, thresh_GB)
    # result1=res1.astype(np.uint8)
    # img2 = cv2.imread("gun1.bmp")
    # rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # res2 = skin_segmentation(rgb2, hist_RG, hist_GB, thresh_RG, thresh_GB)
    # result2=res2.astype(np.uint8)
    # img3 = cv2.imread("pointer1.bmp")
    # rgb3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    # res3 = skin_segmentation(rgb3, hist_RG, hist_GB, thresh_RG, thresh_GB)
    # result3=res3.astype(np.uint8)
    f, axarr = plt.subplots(3,2)
    axarr[0][0].imshow(rgb1)
    axarr[0][1].imshow(result1)
    axarr[1][0].imshow(rgb2)
    axarr[1][1].imshow(result2)
    axarr[2][0].imshow(rgb3)
    axarr[2][1].imshow(result3)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    main()