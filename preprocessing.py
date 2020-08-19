import cv2
import numpy as np
from matplotlib import pyplot as plt


def display_image(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_image(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def rescale(src, dim=(128, 128)):
    dst = cv2.resize(src, dim, interpolation=cv2.INTER_LINEAR)
    return dst


def map_color_space(src, max_val=255):
    b, g, r = cv2.split(img)
    c_1 = (b + g + r)/3
    c_2 = (r + (max_val - b))/2
    c_3 = (r + 2 * (max_val - g) + b)/4
    dst = cv2.merge((c_1, c_2, c_3))
    return dst


if __name__ == '__main__':
    img = cv2.imread('resources/image.jpg', cv2.IMREAD_COLOR)
    dest = rescale(img)
    dest_2 = map_color_space(img)
    plot_image(dest_2)
    cv2.imwrite('messigray.png', dest_2)


