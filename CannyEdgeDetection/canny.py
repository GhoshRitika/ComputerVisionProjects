import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def kernel(size, sigma):
    k=np.zeros((size,size))
    c=size//2
    normal=2.0*np.pi*(sigma**2)
    for i in range(size):
        for j in range(size):
            row = i-c
            col = j-c
            k[i, j]=np.exp(-((row**2 + col**2)/(2.0*(sigma**2))))/normal
    return k


def GaussSmoothing(I, N, Sigma):
    G_k = kernel(N, Sigma)
    S = signal.convolve2d(I, G_k)
    return S

def sobel(I, method):
    if method == 'sobel':
        K_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        K_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    elif method == 'robert':
        K_x = np.array([[1, 0], [0, -1]])
        K_y = np.array([[0, -1], [1, 0]])
    elif method == 'prewitt':
        K_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        K_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    I_x = signal.convolve2d(K_x, I)
    I_y = signal.convolve2d(K_y, I)
    Grad=np.zeros((I.shape))
    dir=np.zeros((I.shape))
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            Grad[i, j]=(I_x[i, j]**2 + I_y[i, j]**2)**0.5
            dir[i, j]=np.arctan2(I_y[i, j], I_y[i, j])
    Grad=(Grad/Grad.max())*255

    return (Grad, dir)

def NMS(mag, theta):
    (h, w)=mag.shape
    R=np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            ang=theta[i,j]*180.0/np.pi
            if(ang<0):
                #convert to 0-360
                ang+=180
            LUT=[(i,j+1), (i-1, j+1), (i-1, j), (i-1, j-1), (i, j-1), (i+1, j-1), (i+1, j),(i+1, j+1)]
            p1=255
            p2=255
            #0deg
            if(0<=ang<22.5 or 157.5<=ang<=180):
                p1=LUT[0]
                p2=LUT[4]
            #45deg
            elif(22.5<=ang<67.5):
                p1=LUT[1]
                p2=LUT[5]
            elif(67.5<=ang<112.5):
                p1=LUT[2]
                p2=LUT[6]
            elif(112.5<=ang<157.5):
                p1=LUT[3]
                p2=LUT[7]
            
            if(p1[0]<0 or p2[0]<0 or p1[0]>h-1 or p2[0]>h-1 or p1[1]<0 or p2[1]<0 or p1[1]>w-1 or p2[1]>w-1):
                R[i, j]=mag[i, j]
            elif(mag[i,j]<mag[p1[0],p1[1]] or mag[i,j]<mag[p2[0],p2[1]]):
                R[i, j]=0
            else:
                R[i, j]=mag[i, j]
    return R

def FindThreshold(Mag, percentageOfNonEdge):
    hist, bins = np.histogram(Mag.flatten(), 256)
    #calculate the cumulative histogram for the above
    cumu = np.cumsum(hist)
    #normalizing the cumulative sum from 0-255
    n_cum = cumu/cumu.max()
    t_high=np.argmax(n_cum>percentageOfNonEdge)
    t_low=0.5*t_high
    return (t_low, t_high)

def EdgeLinking(mag, T_low, T_high):
    row, col = mag.shape
    mag_copy=mag.copy()
    w_val = 10
    s_val = 255
    strong = np.zeros((row, col))
    weak = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if mag[i, j] >= T_high:
                strong[i, j] = s_val
            if mag[i, j] >= T_low:
                weak[i, j] = w_val

    for i in range(1, row-1):
        for j in range(1, col-1):
            LUT=[(i,j+1), (i-1, j+1), (i-1, j), (i-1, j-1), (i, j-1), (i+1, j-1), (i+1, j),(i+1, j+1)]
            if weak[i,j] == w_val:
                for k in range(8):
                    p=LUT[k]
                    if(strong[p[0], p[1]]==s_val):
                        mag_copy[i, j]=s_val
                    else:
                        mag_copy[i, j]=0
    return weak, strong, mag_copy


def main():
    img = cv2.imread("lena.bmp")
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Smoothed=GaussSmoothing(I, 3, 1.0)
    Mag, theta=sobel(Smoothed, 'prewitt')
    non = NMS(Mag, theta)
    T_low, T_high=FindThreshold(Mag, 0.8)
    print(T_high)
    weak, strong, edge=EdgeLinking(non, T_low, T_high)
    res1=Smoothed.astype(np.uint8)
    SM = cv2.cvtColor(res1, cv2.COLOR_GRAY2RGB)
    res2=Mag.astype(np.uint8)
    SO= cv2.cvtColor(res2, cv2.COLOR_GRAY2RGB)
    res3=non.astype(np.uint8)
    MAXI= cv2.cvtColor(res3, cv2.COLOR_GRAY2RGB)
    res4=edge.astype(np.uint8)
    canny= cv2.cvtColor(res4, cv2.COLOR_GRAY2RGB)
    res5=weak.astype(np.uint8)
    weak_canny= cv2.cvtColor(res5, cv2.COLOR_GRAY2RGB)
    res6=strong.astype(np.uint8)
    strong_canny= cv2.cvtColor(res6, cv2.COLOR_GRAY2RGB)
    plt.figure(1)
    plt.title('Using Robert Method')
    plt.subplot(2,4,1)
    plt.title('Original')
    plt.imshow(I, cmap='gray')
    plt.subplot(2,4,2)
    plt.title('Gaussian Smoothed')
    plt.imshow(SM)
    plt.subplot(2,4,3)
    plt.title('Gradient Magnitude')
    plt.imshow(SO)
    plt.subplot(2,4,4)
    plt.title('Nonmaxima Suppressed')
    plt.imshow(MAXI)
    plt.subplot(2,4,5)
    plt.title('Weak edge')
    plt.imshow(weak_canny)
    plt.subplot(2,4,6)
    plt.title('Strong edge')
    plt.imshow(strong_canny)
    plt.subplot(2,4,7)
    plt.title('Edge Linked')
    plt.imshow(canny)
    plt.show()

if __name__ == '__main__': 
    main()
