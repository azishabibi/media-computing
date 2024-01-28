import cv2
import numpy as np
import jittor as jt
from jittor import nn
from PIL import Image
import pdb
# Load the mask image in grayscale mode to ensure it's in one channel
mask_path = './data/completion/input1_mask.jpg'
mask = cv2.imread(mask_path, 0)

# Convert the mask to a binary image
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
binary_mask=255-binary_mask
#import pdb;pdb.set_trace()
# Define the kernel size for dilation
kernel_size = 80  # The size of the dilation as specified
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# Dilate the binary mask
dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
cv2.imwrite('dilated_mask.jpg',dilated_mask)
# Create an image where the dilated areas are white (255) and the rest are black (0)
# This is achieved by XOR operation between the dilated mask and the original binary mask
expanded_area = cv2.bitwise_xor(binary_mask, dilated_mask)

# Save the image of the expanded area
expanded_area_path = 'expanded_area.jpg'
cv2.imwrite(expanded_area_path, expanded_area)

original_image_path = './data/completion/input1.jpg'  # Replace with your image path
original_image = cv2.imread(original_image_path)
if original_image.shape[:2] == dilated_mask.shape:
    # Extract the region of interest from the original image using the dilated mask
    # The dilated mask will serve as a binary mask where the white areas (value 255)
    # specify the pixels to extract from the original image
    region_of_interest = cv2.bitwise_and(original_image, original_image, mask=dilated_mask)

    # Save the extracted region to a file
    extracted_region_path = 'extracted_region.jpg'  # Replace with the desired output path
    cv2.imwrite(extracted_region_path, region_of_interest)
else:
    print("The dimensions of the original image and the dilated mask do not match.")
contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour is the one encompassing the white area
largest_contour = max(contours, key=cv2.contourArea)

# Get the bounding rectangle for the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Draw the bounding rectangle on the original image
#cv2.rectangle(dilated_mask, (x, y), (x+w, y+h), (255, 0, 0), 2)
expanded_area = expanded_area[y:y+h, x:x+w]
region_of_interest=region_of_interest[y:y+h, x:x+w]
cv2.imwrite(extracted_region_path, region_of_interest)
# Save the result or display it
cv2.imwrite('output_image.png', expanded_area)
candiate_path='./data/completion/input1_patch.jpg'
candidate_image=cv2.imread(candiate_path, 0)
region_of_interest_jt = jt.array(dilated_mask)
candidate_image_jt = jt.array(candidate_image)
def compute_l2_error(A_sub, B):
    # Ensure A_sub and B are floating-point tensors
    A_sub = jt.array(A_sub).float32()
    B = jt.array(B).float32()
    import pdb;pdb.set_trace()
    # Check if the batch dimension is present, if not, add it
    # if A_sub.ndim == 3:
    #     A_sub = A_sub.unsqueeze(0)  # Add batch dimension
    # if B.ndim == 2:
    #     B = B.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # # Reorder dimensions to [batch, channels, height, width]
    A_sub = A_sub.unsqueeze(0).unsqueeze(0)
    B = B.unsqueeze(0).unsqueeze(0)
    A_sub = A_sub.permute(0, 1, 2, 3)
    B = B.permute(0, 1, 2, 3)

    # Define the Conv2d layer
    kernel_height, kernel_width = int(A_sub.shape[2]), int(A_sub.shape[3])
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_height, kernel_width), stride=1, padding=0, bias=False)

    # Use A_sub as the weight for the convolution operation
    A_sub = A_sub.stop_grad()
    conv.weight = A_sub
    conv.weight.stop_grad()

    # Perform the convolution operation
    result = conv(B)

    # Compute L2 error
    print(result.shape)
    print(A_sub.shape)
    l2_errors = jt.zeros(result.shape[2:])

    # Calculate L2 error for each position
    for i in range(result.shape[2]):
        for j in range(result.shape[3]):
            # Extract the corresponding patch from B
            patch = B[:, :, i:i+kernel_height, j:j+kernel_width]

            # Calculate the L2 error
            error = (patch - A_sub) ** 2
            l2_errors[i, j] = error.sum()

    # Find the position with minimum error
    min_error_index = jt.argmin(l2_errors)
    min_error_position = np.unravel_index(min_error_index.data, l2_errors.shape)


    return min_error_position

# def conv(crop_np, crop_scale, comp_im, vici_np):
#     nh, nw = crop_np.shape[1] * crop_scale, crop_np.shape[0] * crop_scale
#     nh, nw = int(nh), int(nw)
#     crop_im = Image.fromarray(np.uint8(crop_np)).resize((nh, nw))
#     crop_np = np.asarray(crop_im)
#     vici_np = vici_np.repeat(3, axis=2)
#     vici_im = Image.fromarray(np.uint8(vici_np)).resize((nh, nw))
#     vici_np = np.asarray(vici_im)
#     comp_np = np.asarray(comp_im)
#     [h1, w1, _] = comp_np.shape
#     [h2, w2, _] = crop_np.shape
#     if h2 >= h1 or w2 >= w1:
#         return -1, -1, 1e18
#     comp_conv_jt = jt.array(comp_np, dtype=jt.float32).reindex(
#             [h1 - h2 + 1, w1 - w2 + 1, h2, w2, 3],
#             ['i0 + i2', 'i1 + i3', 'i4'])
#     crop_conv_jt = jt.array(crop_np, dtype=jt.float32).broadcast_var(
#             comp_conv_jt)
#     vici_conv_jt = jt.array(vici_np, dtype=jt.float32).broadcast_var(
#             comp_conv_jt)
#     error_jt = (comp_conv_jt * vici_conv_jt - crop_conv_jt) ** 2
#     error_jt = error_jt.sum([2, 3, 4])
#     error_np = error_jt.fetch_sync()
#     (x, y) = np.unravel_index(error_np.argmin(), error_np.shape)
#     return x, y, error_np[x, y]


# def select_comp(crop_scale_min, crop_scale_max, comp_im, crop_np, vici_np):
#     best_err = 1e18
#     bx, by, bs = None, None, None
#     for crop_scale in np.arange(crop_scale_min, crop_scale_max, 0.1):
#         x, y, err = conv(crop_np, crop_scale, comp_im, vici_np)
#         if err < best_err:
#             bx, by, bs = x, y, crop_scale
#             best_err = err
#     nh, nw = crop_np.shape[1] * bs, crop_np.shape[0] * bs
#     nh, nw = int(nh), int(nw)
#     comp_np = np.asarray(comp_im)
#     comp_np = comp_np[bx:bx + nw, by:by + nh]
#     comp_im = Image.fromarray(np.uint8(comp_np)).resize(
#             (crop_np.shape[1], crop_np.shape[0]))
#     return np.asarray(comp_im)

# # Now call the compute_l2_error function with the Jittor tensors
# comp_np = select_comp(
#             args.crop_scale_min, args.crop_scale_max,
#             candidate_image_jt, crop_np, vici_np)
min_error_position = 0#compute_l2_error(region_of_interest_jt, candidate_image_jt)
print(min_error_position)

# 首先确定 `min_error_position` 对应的区域大小
sub_image_height, sub_image_width = region_of_interest_jt.shape[2], region_of_interest_jt.shape[3]

# 创建一个与 candidate_image 尺寸相同的全黑图像
black_image = np.zeros_like(candidate_image)

# 计算 min_error_position 对应的起始坐标
# 假设 min_error_position 是一个表示位置的索引（例如，在一维化的图像中的索引）
start_y = min_error_position // candidate_image_jt.shape[2]
start_x = min_error_position % candidate_image_jt.shape[3]

# 将 candidate_image 中对应的像素复制到全黑图像中
black_image[start_y:start_y+sub_image_height, start_x:start_x+sub_image_width] = candidate_image[start_y:start_y+sub_image_height, start_x:start_x+sub_image_width]

# 保存修改后的图像
output_path = 'output_with_candidate_pixels.jpg'
cv2.imwrite(output_path, black_image)
