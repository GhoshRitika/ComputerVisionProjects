import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def SSD(I, T):
    I_ht, I_w = I.shape[0], I.shape[1]
    T_ht, T_w = T.shape[0],T.shape[1]
    scores=np.zeros((I_ht-T_ht, I_w-T_w))
    for i in range(0, I_ht-T_ht):
        for j in range(0, I_w-T_w):
            D=(I[i:i+T_ht, j:j+T_w] - T)**2
            scores[i, j]=np.sum(D)
    matched = np.where(scores==np.min(scores))
    return (matched[0][0],matched[1][0])

def CC(I, T):
    I_ht, I_w = I.shape[0], I.shape[1]
    T_ht, T_w = T.shape[0],T.shape[1]
    scores=np.zeros((I_ht-T_ht, I_w-T_w))
    for i in range(0, I_ht-T_ht):
        for j in range(0, I_w-T_w):
            C=(I[i:i+T_ht, j:j+T_w]) * (T)
            scores[i, j]=np.sum(C)
    matched = np.where(scores==np.max(scores))
    return (matched[0][0],matched[1][0])


def NCC(I, T):
    I_ht, I_w = I.shape[0], I.shape[1]
    T_ht, T_w = T.shape[0],T.shape[1]
    scores=np.zeros((I_ht-T_ht, I_w-T_w))
    for i in range(0, I_ht-T_ht):
        for j in range(0, I_w-T_w):
            I_avg=np.mean(I)
            Icap=np.ones((T_ht, T_w))*I_avg
            T_avg=np.mean(T)
            Tcap=np.ones((T_ht, T_w))*T_avg
            Ihat=I[i:i+T_ht, j:j+T_w]-Icap
            That=T-Tcap
            C=Ihat*That
            scores[i, j]=np.sum(C)/(np.sum(Ihat**2)*np.sum(That**2))**0.5
    matched = np.where(scores==np.max(scores))
    return (matched[0][0],matched[1][0])

def NCC_oc(I, T):
    I_ht, I_w = I.shape[0], I.shape[1]
    T_ht, T_w = T.shape[0],T.shape[1]
    scores=np.zeros((I_ht-T_ht, I_w-T_w))
    for i in range(0, I_ht-T_ht):
        for j in range(0, I_w-T_w):
            I_avg=np.mean(I)
            # Icap=np.ones((T_ht, T_w))*I_avg
            T_avg=np.mean(T)
            # Tcap=np.ones((T_ht, T_w))*T_avg
            Ihat=I[i:i+T_ht, j:j+T_w]-I_avg
            That=T-T_avg
            C=Ihat*That
            scores[i, j]=np.sum(C)/(np.sum(Ihat**2)*np.sum(That**2))**0.5
    matched = np.where(scores==np.max(scores))
    return (matched[0][0],matched[1][0])

def Template_matching(method, T, bounds):
    fps = 15
    all_images = glob.glob('image_girl/*.jpg')
    first_frame=cv2.imread(all_images[0])
    img_ht, img_w, _=first_frame.shape
    ht, w=T.shape
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    if method == 'ssd':
        func=SSD
        out=cv2.VideoWriter('SSD_result.mp4', fourcc, fps, (img_w,img_ht))
        dest='SSD'
    elif method == 'cc':
        func=CC
        out=cv2.VideoWriter('CC_result.mp4', fourcc, fps, (img_w,img_ht))
        dest='CC'
    elif method == 'ncc':
        func=NCC
        out=cv2.VideoWriter('NCC_result.mp4', fourcc, fps, (img_w,img_ht))
        dest='NCC'
    # matched=(bounds[1], bounds[0])
    # cv2.rectangle(first_frame, (matched[1], matched[0]), (matched[1] + w, matched[0] + ht), (0, 255, 0), 2)
    # print('%s/%s' %(dest,all_images[0]))
    # cv2.imwrite('%s/%s' %(dest,all_images[0]), first_frame)
    # out.write(first_frame)
    for name in sorted(all_images):
        i=cv2.imread(name)
        gray=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        # cv2.rectangle(i, (matched[1], matched[0]), (matched[1] + w, matched[0] + ht), (0, 255, 0), 2)
        matched=func(gray, T)
        cv2.rectangle(i, (matched[1], matched[0]), (matched[1] + w, matched[0] + ht), (0, 255, 0), 2)
        # T=gray[matched[0]:matched[0]+ht, matched[1]:matched[1]+w]
        # print('%s/%s' %(dest,name))
        cv2.imwrite('%s/%s' %(dest,name), i)
        out.write(i)
    out.release()

def Occlusion(T, bounds):
    fps = 15
    all_images = glob.glob('image_girl/*.jpg')
    first_frame=cv2.imread(all_images[0])
    img_ht, img_w, _=first_frame.shape
    ht, w,_=T.shape
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter('OCCLUSION_result.mp4', fourcc, fps, (img_w,img_ht))
    dest='OCCLUSION'
    for name in sorted(all_images):
        i=cv2.imread(name)
        gray=cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
        match1=SSD(gray, T)
        match2=NCC_oc(gray, T)
        matched=(int((match1[0]+match2[0])/2), int((match1[1]+match2[1])/2))
        # print(matched)
        cv2.rectangle(i, (matched[1], matched[0]), (matched[1] + w, matched[0] + ht), (0, 255, 0), 2)
        # print('%s/%s' %(dest,name))
        cv2.imwrite('%s/%s' %(dest,name), i)
        out.write(i)
    out.release()

def main():
    all_images = glob.glob('image_girl/*.jpg')
    first_frame=cv2.imread(all_images[0])
    corners = cv2.selectROI(first_frame)
    cv2.destroyAllWindows()
    x, y, w, h = corners
    # cv2.rectangle(first_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    ff_gray=cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    ff_hsv=cv2.cvtColor(first_frame,cv2.COLOR_BGR2HSV)
    template=ff_gray[y:y+h, x:x+w]
    template_oc=ff_hsv[y:y+h, x:x+w]
    Template_matching('ssd', template, corners)
    Template_matching('cc', template, corners)
    Template_matching('ncc', template, corners)
    Occlusion(template_oc, corners)

if __name__ == '__main__': 
    main()
