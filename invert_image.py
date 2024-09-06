import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib  import Path
import re
import os

# from google.colab.patches import cv2_imshow

def show_images(*images, wait=True):
    """Display multiple images with one line of code"""

    for image in images:
        cv2_imshow(image)

    # if wait:
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

def plot_color_hist(img):
    # Histogram plotting of the imagr
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img],
                            [i], None,
                            [256],
                            [0, 256])
        plt.plot(histr, color = col)
        plt.xlim([0, 256])

    plt.show()

def invert_bgr(img_bgr):
    ''' inplace = true'''

    # get height and width of the image
    height, width, _ = img_bgr.shape
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            # Get the pixel value
            pixel = img_bgr[i, j]

            # Negate each channel by
            # subtracting it from 255
            pixel[0] = 255 - pixel[0]
            pixel[1] = 255 - pixel[1]
            pixel[2] = 255 - pixel[2]

            img_bgr[i, j] = pixel
    return pixel

def convert_bw(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # thresh = 137
    print(thresh)
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    return im_bw

def invert_bw(img_bw):
    ''' inplace = true'''
    # get height and width of the image
    height, width = img_bw.shape
    for i in range(0, height - 1):
        for j in range(0, width - 1):

            # Get the pixel value
            img_bw[i, j] = 255 - img_bw[i, j]
    return img_bw

def remove_colored_bg(img):
    img_bw = convert_bw(img)
    invert_bw(img_bw)

    return img_bw

def plot_images(*imgs):
    for img in imgs:
        plt.imshow(img)
        plt.show()

def save_img(img, dst):
    if isinstance(dst, Path):
        dst = str(dst.absolute())

    cv2.imwrite(dst, img)



def key_num(x):
    return int(re.search('1.2_(\d*)', x).group(1))

# l = os.listdir('/content/drive/MyDrive/Sawn 1.2/black_white')
# l.sort(key= key_num)

def collage_pic(src:str, file_list):
    src = Path(src)
    src.joinpath('collaged').mkdir(exist_ok=True)
    src = str(Path(src).absolute())

    for i in range(0, len(file_list)-3 or 1, 3):
        print(src)
        l=[]
        for j in file_list[i:i+3]:
            img = cv2.imread(f"{src}/{j}", 1)
            l.append(img)

        collage = np.vstack(l)
        save_img(collage, f"{src}/collaged/{j}")

# collage_pic('.', ['a.jpg']*3)



"""# work place"""

src = '/content/drive/MyDrive/Sawn 1.2'

src = Path(src)
dst = src.joinpath('black_white')
dst.mkdir(exist_ok=True)
img_ext = ['jpg', 'png']

for img_path in os.listdir(src):
    ext = img_path[img_path.rfind('.')+1:]
    img_des = dst.joinpath(img_path)
    img_path = src.joinpath(img_path)
    # print(ext)

    if img_path.is_file() and ext in img_ext:
        print(img_path)
        img_bgr = cv2.imread(str(img_path.absolute()), 1)
        # plt.imshow(img_bgr)
        # plt.show()
        img_bw = remove_colored_bg(img_bgr)
        # plt.imshow(img_bw)
        # plt.show()

        # print(img_des)
        cv2.imwrite(str(img_des.absolute()), img_bw)
    # break

# Read an image
img = 'a.jpg'
img_bgr = cv2.imread(img, 1)
# plt.imshow(img_bgr)
# plt.show()
a=remove_colored_bg(img_bgr)

show_images(img_bgr)

show_images(a)

# !cp -r "/content/drive/MyDrive/Sawn 1.2" .
!ls -sh "Sawn_1.2/black_white"

cv2.imwrite('sample_data/c.jpg', img_bw)







"""# image upscale"""

im_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

im_gray.shape
a=invert_bw(im_gray)

# show_images(im_gray)
show_images(a)

!wget "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x3.pb"

sr = cv2.dnn_superres.DnnSuperResImpl_create()

path = "ESPCN_x3.pb"
sr.readModel(path)
sr.setModel("espcn",3)

result = sr.upsample(img)

# Resized image
resized = cv2.resize(img,dsize=None,fx=3,fy=3)

plt.figure(figsize=(60,25))
plt.subplot(1,3,1)
# Original image
plt.imshow(img[:,::-1])
plt.subplot(1,3,2)
# SR upscaled
plt.imshow(result[:,::-1])
plt.subplot(1,3,3)
# OpenCV upscaled
plt.imshow(resized[:,::-1])
plt.show()

img = a

a = remove_colored_bg(img_bgr)

show_images(a, cv2.resize(a,dsize=None,fx=3,fy=3))



"""# Extra"""

plot_color_hist(img_bgr)

show_images(img_bw)

img_bw

# plt.imshow(img_bgr)
# plt.imshow(img_bgr)
# show_images(im_gray)
# show_images(im_bw)





backtorgb = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2RGB)
show_images(backtorgb)

(backtorgb.shape)

# backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
xor_img = cv2.bitwise_or(backtorgb, img_bgr)
show_images(xor_img)

import numpy as np
(np.unique(img_bgr))

# method 2
import cv2
import matplotlib.pyplot as plt


# Read an image
img_bgr = cv2.imread('a.jpg', 1)
plt.imshow(img_bgr)
plt.show()


# Histogram plotting of original image
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img_bgr],
                         [i], None,
                         [256],
                         [0, 256])

    plt.plot(histr, color = col)

    # Limit X - axis to 256
    plt.xlim([0, 256])

plt.show()


# Negate the original image
img_neg = 1 - img_bgr
plt.imshow(img_neg)
plt.show()

# Histogram plotting of
# negative transformed image
color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histr = cv2.calcHist([img_neg],
                         [i], None,
                         [256],
                         [0, 256])

    plt.plot(histr, color = col)
    plt.xlim([0, 256])

plt.show()











# Import the library OpenCV

import cv2


# Import the image
file_name = "gfg_black.png"

# Read the image
src = cv2.imread(file_name, 1)

# Convert image to image gray
tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# Applying thresholding technique
_, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)

# Using cv2.split() to split channels
# of coloured image
b, g, r = cv2.split(src)

# Making list of Red, Green, Blue
# Channels and alpha
rgba = [b, g, r, alpha]

# Using cv2.merge() to merge rgba
# into a coloured/multi-channeled image
dst = cv2.merge(rgba, 4)

# Writing and saving to a new image
cv2.imwrite("gfg_white.png", dst)

dst = cv2.merge(rgba, 4)


# Writing and saving to a new image

cv2.imwrite("gfg_white.png", dst)



