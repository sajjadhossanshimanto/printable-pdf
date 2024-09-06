
# Read an image
img = 'a.jpg'
img_bgr = cv2.imread(img, 1)

a=remove_colored_bg(img_bgr)
show_images(img_bgr)
show_images(a)




""" image upscale """
#%%
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

