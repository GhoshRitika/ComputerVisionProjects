import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as lin

def Hough(img, canny, thresh):
    row, col = canny.shape

    black=np.zeros((col,row))
    r_size = (row**2 + col**2)**0.5
    dr= 2*r_size/180
    rho= np.linspace(-r_size, r_size, int(2*r_size / dr) + 1)
    votes= np.zeros((len(rho), len(rho)))

    theta = np.linspace(0, 180, 180 + 1)
    cos=np.cos(theta*(np.pi/180.0))
    sin=np.sin(theta*(np.pi/180.0))

    fig, axs = plt.subplots(2, 2)
    plots = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

    plots[0].imshow(img)
    plots[0].set_title('Original')
    plots[1].imshow(canny, cmap='gray')
    plots[1].set_title('Canny Edge')
    plots[2].set_facecolor('black')
    plots[3].imshow(img)

    # Parameter space loops
    for i in range(row):
        for j in range(col):
            if canny[i][j] != 0:
                x=i - row/2
                y=j - col/2
                m = []
                c = []
                for t in range(len(theta)):
                    rh = (y * cos[t]) + (x * sin[t])
                    th = theta[t]
                    val = np.argmin(np.abs(rho - rh))
                    votes[val][t] += 1
                    c.append(rh)
                    m.append(th)
                plots[2].plot(m, c, color='white', alpha=0.05)

    plots[2].set_title('Parameter Space')

    # Line determining loops
    for i in range(len(rho)):
        for j in range(len(rho)):
            if votes[i][j] > thresh:
                r = rho[i]
                th = theta[j]
                p1 = np.cos(th*(np.pi/180.0))
                p2 = np.sin(th*(np.pi/180.0))
                x0 = (p1*r) + col/2
                y0 = (p2*r) + row/2
                x1 = int(x0 + 100*-p2)
                y1 = int(y0 + 100*p1)
                x2 = int(x0 - 100*-p2)
                y2 = int(y0 - 100*p1)
                plots[3].add_line(lin.Line2D([x1, x2], [y1, y2], color='red'))

    plots[3].set_title('Lines Detected')

    plt.show()

def main():
    image = cv2.imread('test2.bmp')
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray, (5,5), 0.5)
    edge = cv2.Canny(blur, 50, 150, 5)
    Hough(image, edge,100)


if __name__ == '__main__': 
    main()
