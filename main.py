import cv2
import PIL
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imshow
from numpy import asarray
import skimage
import imutils
from scipy.fftpack import dct, idct
from scipy import fft
from scipy import fftpack
import scipy
from scipy import misc

# ******* PLOTTING THE ORIGINAL IMAGE ********
plt.rcParams["figure.figsize"] = [7, 7]
plt.rcParams["figure.autolayout"] = True
image = plt.imread("flamingo.jpg")
original = cv2.imread("flamingo.jpg")

plt.subplots()
plt.title("Image of a flamingo")
plt.imshow(image, extent=[0, 300, 0, 300])
plt.plot()
plt.show()

M, N = 2236, 2236

# LOW PASS FILTER
lpf = np.zeros((M, N))
Do = 600
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        if D <= Do:
            lpf[u, v] = 1
        else:
            lpf[u, v] = 0

plt.subplots()
plt.imshow(lpf)
plt.title("Ideal low pass filter")
plt.plot()
plt.show()

# HIGH PASS FILTER
hpf = np.zeros((M, N))
Do = 600
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        if D >= Do:
            hpf[u, v] = 1
        else:
            hpf[u, v] = 0

plt.subplots()
plt.imshow(hpf)
plt.title("Ideal high pass filter")
plt.plot()
plt.show()

# CONVERTING THE IMAGE TO GRAYSCALE TO EQUALIZE DIMENSIONS WITH FILTERS
real_image = imread('flamingo.jpg')
imgGray = cv2.imread('flamingo.jpg', 0)
original = np.reshape(imgGray, (2236, 2236))
plt.subplots()
plt.imshow(imgGray, cmap='gray')
plt.title("Grayscale")
plt.show()


# # CREATING A 3X3 IDEAL HIGH PASS FILTER
high_pass_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
image_from_hpf = cv2.filter2D(original, -1, high_pass_filter)
plt.subplots()
plt.imshow(image_from_hpf, cmap='gray')
plt.title("High passed image")
plt.plot()
plt.show()

# CREATING A 3X3 IDEAL LOW PASS FILTER
low_pass_filter = np.array([[1/32, 1/16, 1/32], [1/16, 1/96, 1/16], [1/32, 1/16, 1/32]])
image_from_lpf = cv2.filter2D(original, -1, low_pass_filter)
plt.subplots()
plt.imshow(image_from_lpf, cmap='gray')
plt.title("Low passed image")
plt.plot()
plt.show()

# FINDING THE FREQUENCY SPECTRUM OF ORIGINAL IMAGE
original_image_fourier = np.fft.fftshift(np.fft.fft2(original))
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(np.log(abs(original_image_fourier)), cmap='gray')
plt.title("Original image frequency spectrum")
plt.show()

# FINDING THE FREQUENCY SPECTRUM OF IMAGE PASSED FROM HIGH PASS FILTER
hpf_image_fourier = np.fft.fftshift(np.fft.fft2(image_from_hpf))
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(np.log(abs(hpf_image_fourier)), cmap='gray')
plt.title("High passed image frequency spectrum")
plt.show()

# FINDING THE FREQUENCY SPECTRUM OF IMAGE PASSED FROM LOW PASS FILTER
lpf_image_fourier = np.fft.fftshift(np.fft.fft2(image_from_lpf))
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(np.log(abs(lpf_image_fourier)), cmap='gray')
plt.title("Low passed image frequency spectrum")
plt.show()



# **************** module 2 ****************
# TASK 5
# FINDING PYRAMIDS OF THE IMAGE BY 2X
img_copy = imgGray.copy()
# 2x
plt.subplot(2, 2, 1)
img_copy = cv2.pyrDown(img_copy)
min_2x =np.floor(np.size(img_copy))
print(min_2x)
plt.title("Original image")
plt.imshow(imgGray, cmap='gray')
plt.subplot(2,2,2)
plt.title("Reduced by 2x")
plt.imshow(img_copy, cmap='gray')
# 8x
plt.subplot(2, 2, 3)
img_copy = cv2.pyrDown(img_copy)
img_copy = cv2.pyrDown(img_copy)
plt.title("Reduced by 8x")
plt.imshow(img_copy, cmap='gray')
# 16x
plt.subplot(2, 2, 4)
img_copy = cv2.pyrDown(img_copy)
plt.title("Reduced by 16x")
plt.imshow(img_copy,cmap='gray')
# main plot
plt.suptitle("Image pyramids (Gaussian Blur)")
plt.show()



# TASK 6
'''
# IN REPORT DOCUMENT
'''

# TASK 7
'''
# IN REPORT DOCUMENT
'''

# TASK 8
aliased_image = cv2.imread("aliasing_image.png")
plt.suptitle("Disobeying Nyquist theorem")
plt.subplot(212), plt.imshow(imgGray, cmap='gray'), plt.title("Original image")
plt.subplot(211), plt.imshow(aliased_image), plt.title("Aliased down-sampled image")
plt.show()

# TASK 9
# IN REPORT DOCUMENT
# new_image = cv2.resize(np.log(abs(original_image_fourier)), (500, 500))
# ************* MODULE 3 ***************
# TASK 1
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)

# FINDING THE FREQUENCY SPECTRUM OF ORIGINAL IMAGE
original_image_fft = np.fft.fftshift(np.fft.fft2(imgGray))
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(np.log(abs(original_image_fft)), cmap='gray')
plt.title("Original image FFT")
plt.show()
# INVERSE FOURIER TRANSFORM
original_image_ifft = np.fft.ifft2(original_image_fft)
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(abs(original_image_ifft), cmap='gray')
plt.title("Original image inverse FFT")
plt.show()
# PASSING THE FFT IMAGE THROUGH LPF
fft_image_from_lpf = cv2.filter2D(abs(original_image_ifft), -1, low_pass_filter)
plt.subplot(121) ,plt.imshow(fft_image_from_lpf, cmap='gray'), plt.title("Low passed fft image")
# PASSING THE FFT IMAGE THROUGH HPF
fft_image_from_hpf = cv2.filter2D(abs(original_image_ifft), -1, high_pass_filter)
plt.subplot(122), plt.imshow(fft_image_from_hpf, cmap='gray'), plt.title("High passed fft image")
plt.show()


# TASK 2
# NEAREST NEIGHBOR INTERPOLATION
near_img = cv2.resize(imgGray, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
plt.imshow(near_img, cmap='gray')
plt.title("Near interpolation (Zoomed 3X)", fontsize=10)
plt.show()

# BI-LINEAR INTERPOLATION
bilinear_img = cv2.resize(imgGray,None, fx = 3, fy =3, interpolation = cv2.INTER_LINEAR)
plt.imshow(bilinear_img, cmap='gray')
plt.title("Bilinear interpolation (Zoomed 3X)", fontsize=10)
plt.show()

# BICUBIC INTERPOLATION
bicubic_img = cv2.resize(imgGray,None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
plt.imshow(bicubic_img, cmap='gray')
plt.title("Bicubic interpolation (Zoomed 3X)", fontsize=10)
plt.show()

# LANCZOS INTERPOLATION
lanczos_img = cv2.resize(imgGray,None, fx = 3, fy = 3, interpolation = cv2.INTER_LANCZOS4)
plt.imshow(lanczos_img, cmap='gray')
plt.title("Lanczos4 interpolation (Zoomed 3X)", fontsize=10)
plt.show()


# TASK 3
import cv2
import numpy as np
real_image = cv2.imread('flamingo.jpg', 0)
# FINDING DCT
floating_image = np.float32(real_image)
floating_image_cosine_transform = cv2.dct(floating_image, cv2.DCT_INVERSE)
# FINDING INVERSE DCT
inverse_DCT = cv2.idct(floating_image_cosine_transform)
inverse_DCT = np.uint8(inverse_DCT)
# PLOTTING DCT OF IMAGE
cv2.namedWindow('DCT of image', cv2.WINDOW_NORMAL)
floating_image_cosine_transform = cv2.resize(floating_image_cosine_transform, (2200, 2200))
cv2.imshow("DCT of image", floating_image_cosine_transform)
cv2.waitKey(0)
# PLOTTING INVERSE DCT
cv2.namedWindow('inverse DCT of image', cv2.WINDOW_NORMAL)
inverse_DCT = cv2.resize(inverse_DCT, (2200, 2200))
cv2.imshow("inverse DCT of image", inverse_DCT)
cv2.waitKey(0)

# PLOTTING DCT COEFFICIENTS
plt.figure(figsize=(16, 12))
plt.hist(np.log10(np.abs(floating_image_cosine_transform.ravel())), bins=100, color='#282828', alpha=.3, histtype='stepfilled')
plt.xlabel('Amplitude of DCT Coefficients (log-scaled)')
plt.ylabel('Number of DCT Coefficients')
plt.title('Amplitudes of Coefficients')
plt.show()




