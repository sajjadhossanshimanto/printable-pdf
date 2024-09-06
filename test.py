
# Read an image
img = 'a.jpg'
img_bgr = cv2.imread(img, 1)
# plt.imshow(img_bgr)
# plt.show()
a=remove_colored_bg(img_bgr)

show_images(img_bgr)
show_images(a)



cv2.imwrite('sample_data/c.jpg', a)
