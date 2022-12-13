% matplotlib inline
import skimage
print(skimage.__version__)
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PIL.ImageChops import add, subtract, multiply, difference, screen
import PIL.ImageStat as stat
from skimage.io import imread, imsave, imshow, show, imread_collection, imshow_collection
from skimage import color, viewer, exposure, img_as_float, data
from skimage.transform import SimilarityTransform, warp, swirl
from skimage.util import invert, random_noise, montage
import matplotlib.image as mpimg
import matplotlib.pylab as plt
from scipy.ndimage import affine_transform, zoom
from scipy import misc

im = Image.open("../images/parrot.png") 
print(im.width, im.height, im.mode, im.format, type(im))
im.show() 

im = mpimg.imread("../images/hill.png") 
print(im.shape, im.dtype, type(im))
plt.figure(figsize=(10,10))
plt.imshow(im) 
plt.axis('off')
plt.show()

















