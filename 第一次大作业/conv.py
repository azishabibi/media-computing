import jittor as jt
from jittor import nn
import numpy as np
import cv2

# Load the original region (A) and the filling image (B)
A = cv2.imread('extracted_region.jpg') # Placeholder path
B = cv2.imread('./data/completion/input1_patch.jpg') # Placeholder path

# Ensure A and B are in the same dimensions, you might need to resize or pad B
# B = cv2.resize(B, (A.shape[1], A.shape[0]))

# Convert images to float and normalize if needed
A = A.astype(np.float32) / 255.0 # 351 785 3
B = B.astype(np.float32) / 255.0 #(768, 1024, 3)

# Load the mask and prepare it as a kernel
mask = cv2.imread('output_image.png', 0) # Placeholder path (351, 785)
#import pdb;pdb.set_trace()
kernel = mask.astype(np.float32) / 255.0
kernel = kernel.reshape(1, kernel.shape[0], kernel.shape[1]) # Shape: (1, H, W, 1) for grayscale

# Convert numpy arrays to jittor variables
jA = jt.array(A).permute(2, 0, 1).unsqueeze(0) # Convert to NCHW format
jB = jt.array(B).permute(2, 0, 1).unsqueeze(0) # Convert to NCHW format
jKernel = jt.array(kernel).unsqueeze(0) # Shape: (1, 1, H, W)
jKernel = jKernel.repeat(1, 3, 1, 1)

# Disable tuner
jt.flags.enable_tuner = 0

#Define the convolution function using Jittor
def conv(crop_np, crop_scale, comp_im, vici_np):
    # nh, nw = crop_np.shape[1] * crop_scale, crop_np.shape[0] * crop_scale
    # nh, nw = int(nh), int(nw)
    # crop_im = Image.fromarray(np.uint8(crop_np)).resize((nh, nw))
    # crop_np = np.asarray(crop_im)
    import pdb;pdb.set_trace()
    vici_np=np.expand_dims(vici_np, 2)
    vici_np = vici_np.repeat(3, axis=2)
    #(393, 800, 3) (393, 800, 1)
    # vici_im = Image.fromarray(np.uint8(vici_np)).resize((nh, nw))
    # vici_np = np.asarray(vici_im)
    comp_np = np.asarray(comp_im)#(768, 1024, 3)
    [h1, w1, _] = comp_np.shape
    [h2, w2, _] = crop_np.shape
    if h2 >= h1 or w2 >= w1:
        return -1, -1, 1e18
    comp_conv_jt = jt.array(comp_np, dtype=jt.float32).reindex(
            [h1 - h2 + 1, w1 - w2 + 1, h2, w2, 3],
            ['i0 + i2', 'i1 + i3', 'i4'])
    crop_conv_jt = jt.array(crop_np, dtype=jt.float32).broadcast_var(
            comp_conv_jt)
    vici_conv_jt = jt.array(vici_np, dtype=jt.float32).broadcast_var(
            comp_conv_jt)
    error_jt = (comp_conv_jt * vici_conv_jt - crop_conv_jt) ** 2
    error_jt = error_jt.sum([2, 3, 4])
    error_np = error_jt.fetch_sync()
    import pdb;pdb.set_trace()
    (x, y) = np.unravel_index(error_np.argmin(), error_np.shape)
    return x, y, error_np[x, y]

x,y,error=conv(A,1,B,mask)
# Perform convolution
#padding = (jKernel.shape[2] // 2, jKernel.shape[3] // 2)

# conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel.shape[1:], bias=False, groups=3)
# conv2d.weight = jKernel

# # Perform convolution
# jConvolvedB = conv2d(jB)

# Calculate the pixel-wise difference
# print(jA.shape, jConvolvedB.shape)
# import pdb;pdb.set_trace()
# difference = jA - jConvolvedB

# Find the region with the minimum difference
# This is a placeholder for the logic you'd use to find the minimum difference region.
# You might calculate a sum of differences or any other metric that suits your needs.
#min_diff_region = difference.argmin() # Placeholder operation

# Print shapes and display an image

# plt.imshow(jConvolvedB[0, 0, :, :].data) # Uncomment to display the image with matplotlib

# Assuming you want the result as a numpy array
#result = jConvolvedB.numpy()
