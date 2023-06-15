import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv2.imread("palm.bmp")
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (ht, width) = I.shape
    se_s = [(-1,-1),(-1, 0), (-1,1),
            (0, -1), (0,0), (0,1), 
            (1, -1), (1, 0), (1,1)]
    se = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
        (-1, -2), (-1,-1),(-1, 0), (-1,1),(-1, 2),
        (0, -2), (0, -1), (0,0), (0,1), (0, 2),
        (1, -2), (1, -1), (1, 0), (1,1), (1, 2),
        (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]
    E = Erosion(I, se_s)
    D = Dilation(I, se)
    O = Opening(I, se_s)
    C = Closing(I, se)
    B = Boundary(C, se)
    f, axarr = plt.subplots(2,3)
    axarr[0, 0].imshow(I)
    axarr[0, 1].imshow(E)
    axarr[0, 2].imshow(D)
    axarr[1, 0].imshow(O)
    axarr[1, 1].imshow(C)
    axarr[1, 2].imshow(B)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Erosion(I, SE):
    (ht, width) = I.shape
    L = np.zeros((ht, width), dtype=np.uint8)
    for u in range(2,ht-2):
        for v in range(2,width-2):
            flag = False
            for s in SE:
                if I[u+s[0], v+s[1]] == 0:
                    flag = True
            if flag == False:
                L[u, v] = I[u, v]
    return L


def Dilation(I, SE):
    (ht, width) = I.shape
    L = np.zeros((ht, width), dtype=np.uint8)
    for u in range(ht):
        for v in range(width):
            if I[u, v] > 0:
                for s in SE:
                    L[u+s[0], v+s[1]] = I[u, v]
    return L

def Opening(I, SE):
    eroded = Erosion(I, SE)
    result = Dilation(eroded, SE)
    return result

def Closing(I, SE):
    dilated = Dilation(I, SE)
    result = Erosion(dilated, SE)
    return result

def Boundary(I, SE):
    Dilated = Dilation(I, SE)
    closed = Erosion(Dilated, SE)
    eroded = Erosion(I, SE)
    result = closed - eroded
    return result

if __name__ == '__main__': 
    main()