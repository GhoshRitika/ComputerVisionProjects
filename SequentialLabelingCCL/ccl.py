import cv2
import numpy as np

def main():
    img = cv2.imread("gun.bmp")
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (ht, width) = I.shape
    L = np.zeros((ht, width), dtype=np.uint8)
    label_set = set()
    i = 0
    for u in range(ht):
        for v in range(width):
            i = labeling(I, L, u, v, label_set, i)
    sizes = np.zeros(len(label_set))

    for t in range(len(label_set)):
        for u in range(L.shape[0]):
            for v in range(L.shape[1]):
                if(L[u, v] == list(label_set)[t]):
                    sizes[t] +=1
    for t in range(len(label_set)):
        if (sizes[t] < 800):
            for u in range(L.shape[0]):
                for v in range(L.shape[1]):
                    if(L[u, v] == list(label_set)[t]):
                        L[u, v] = 0
    gray = cv2.merge([L, L, L])
    cv2.imshow('image', gray)
    cv2.imwrite('gun_labelled.bmp', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def labeling(I, L, u, v, l_set, label):
    if I[u, v]==255:
        Lu = L[u-1, v] #Upper label
        Ll = L[u, v-1] #Left label
        if Lu == Ll and Lu != 0 and Ll != 0: # not the same label
            L[u, v] = Lu
        elif Lu != Ll and not(Lu and Ll): #if either is 0
            L[u, v] = max(Lu, Ll)
        elif Lu != Ll and Lu>0 and Ll>0: #both
            L[u, v] = min(Lu, Ll)
            E_table(L, Lu, Ll, l_set)
            if (label == max(Lu, Ll)):
                label -=25
        else:
            label+=25
            L[u, v] = label
            l_set.add(label)
    return label

def E_table(L, upper, left, label_set):
    for u in range(L.shape[0]):
        for v in range(L.shape[1]):
            if(L[u, v] == max(upper, left)):
                L[u, v] = min(upper, left)
    if max(upper, left) in label_set:
        label_set.remove(max(upper, left))

if __name__ == '__main__': 
    main()