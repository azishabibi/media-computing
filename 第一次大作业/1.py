import cv2
import numpy as np
import jittor as jt
from jittor import nn
from PIL import Image
import pdb
from conv import conv
from cut import graph_cut
def main(original_image_path,mask_path,patch_image_path):   
    mask = cv2.imread(mask_path, 0)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    binary_mask=255-binary_mask
    kernel_size = 80  # The size of the dilation as specified
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilate the binary mask
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    cv2.imwrite('dilated_mask.jpg',dilated_mask)
    _, binary_mask_sink = cv2.threshold(dilated_mask, 127, 255, cv2.THRESH_BINARY)
    binary_mask_sink=255-binary_mask_sink

    # Detect edges using Canny
    edges_sink = cv2.Canny(binary_mask_sink, 100, 200)

    # Invert edges image: edges become white, the rest becomes black
    sinkcap = cv2.bitwise_not(edges_sink)
    expanded_area = cv2.bitwise_xor(binary_mask, dilated_mask)

    # Save the image of the expanded area
    # expanded_area_path = 'expanded_area.jpg'
    # cv2.imwrite(expanded_area_path, expanded_area)
    original_image = cv2.imread(original_image_path)
    if original_image.shape[:2] == dilated_mask.shape:
        region_of_interest = cv2.bitwise_and(original_image, original_image, mask=expanded_area)

        # # Save the extracted region to a file
        # extracted_region_path = 'extracted_region.jpg'  # Replace with the desired output path
        # cv2.imwrite(extracted_region_path, region_of_interest)
    else:
        print("The dimensions of the original image and the dilated mask do not match.")
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edges = cv2.Canny(binary_mask, 100, 200)

    # Invert edges image: edges become white, the rest becomes black
    sourcecap = cv2.bitwise_not(edges)

    # Assuming the largest contour is the one encompassing the white area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)#0 249 785 351

    # Draw the bounding rectangle on the original image
    #cv2.rectangle(dilated_mask, (x, y), (x+w, y+h), (255, 0, 0), 2)
    mask_small = expanded_area[y:y+h, x:x+w]
    region_of_interest=region_of_interest[y:y+h, x:x+w]
    sourcecap=sourcecap[y:y+h, x:x+w]
    sinkcap=sinkcap[y:y+h, x:x+w]
    print(sinkcap.shape,sourcecap.shape)
    patch_image=cv2.imread(patch_image_path)
    x1,y1,error=conv(region_of_interest,1,patch_image,mask_small)
    patch_image = np.asarray(patch_image)
    h1, w1 = region_of_interest.shape[1], region_of_interest.shape[0]
    patch_image = patch_image[x1:x1 + w1, y1:y1 + h1]#381 785 3
    segments=graph_cut(region_of_interest, patch_image, mask_small)
    segments=segments.astype(np.uint8) * 255 #(351, 785)
    big_patch=255-np.zeros(original_image.shape)#600 800 3
    pdb.set_trace()
    big_patch[y:y+h,x:x+w]=patch_image
    big_mask=255-np.zeros(original_image.shape)
    arr_3d = np.stack([segments] * 3, axis=-1)
    big_mask[y:y+h,x:x+w]=arr_3d

    # 保存或显示结果
    # output_path = 'seg.jpg'
    # cv2.imwrite(output_path, segments)
    #import pdb;pdb.set_trace()
    #cv2.imwrite("sourcecap.jpg",sourcecap)
    #cv2.imwrite("sinkcap.jpg",sinkcap)
    #cv2.imwrite(extracted_region_path, region_of_interest)
    # Save the result or display it
    #cv2.imwrite('output_image.png', expanded_area)

mask_path = './data/completion/input1_mask.jpg'
original_image_path = './data/completion/input1.jpg'
patch_image_path='./data/completion/input1_patch.jpg'
main(original_image_path,mask_path,patch_image_path)