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



def key_num(x: str):
    '''logic: 
        - split the extextion
        - take digits from the rightmost dash ( _ )
    '''
    return int(str(x).rsplit(".", 1)[0].rsplit("_", 1)[-1])
    # return int(re.search('1.2_(\d*)', x).group(1))

def collage_pic(src:str, file_list):
    ''' file_list: user have to filter folder and other file type
     as well as sort then according. as os.listdir returns unordered file list '''
    src = Path(src)
    src.joinpath('collaged').mkdir(exist_ok=True)
    src = str(Path(src).absolute())

    for i in range(0, len(file_list)-3 or 1, 3):
        print(src)
        l=[]
        for j in file_list[i:i+3]:
            img = cv2.imread(f"{src}{sep}{j}", 1)
            l.append(img)

        collage = np.vstack(l)
        save_img(collage, f"{src}{sep}collaged{sep}{j}")

def filter_filetype(folder:str, filter_ext:list) -> list['file name']:
    for file_path in os.listdir(folder):
        ext = img_path[img_path.rfind('.')+1:]
        if ext in filter_ext:
            yield file_path


"""# work place"""

src = '/content/drive/MyDrive/Sawn 1.2'
src = 'c3'

src = Path(src)
sep = src._flavour.sep
dst = src.joinpath('black_white')
dst.mkdir(exist_ok=True)
img_ext = ['jpg', 'png']

l = []
for img_path in os.listdir(src):
    l.append(img_path)# l should only cointain the file name

    ext = img_path[img_path.rfind('.')+1:]
    img_des = dst.joinpath(img_path)
    img_path = src.joinpath(img_path)
    # print(ext)

    if img_path.is_file() and ext in img_ext:
        print(img_path)
        continue# uncomment while just need to collage

        img_bgr = cv2.imread(str(img_path.absolute()), 1)
        
        img_bw = remove_colored_bg(img_bgr)
        
        # print(img_des)
        cv2.imwrite(str(img_des.absolute()), img_bw)
    else:
        l.pop()
    # break


l.sort(key=key_num)
# print(l)
collage_pic(src, l)
