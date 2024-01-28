import cv2
import numpy as np
import jittor as jt
import argparse
import os
import maxflow
import scipy.sparse
import pyamg
if jt.has_cuda:
    jt.flags.use_cuda = 1


def compute_error(target_area, patch_image, mask):
    mask_with_channels = np.expand_dims(mask, axis=2).repeat(3, axis=2)
    patch = jt.array(np.asarray(patch_image), dtype=jt.float32)
    target = jt.array(target_area, dtype=jt.float32)
    mask_ = jt.array(mask_with_channels, dtype=jt.float32)
    patch_height, patch_width, _ = patch.shape
    target_height, target_width, _ = target.shape

    # adjust patch image for conv
    conv_patch = patch.reindex(
        [patch_height - target_height + 1, patch_width - target_width + 1, target_height, target_width, 3],
        ['i0 + i2', 'i1 + i3', 'i4']
    )
    broad_mask = mask_.broadcast_var(conv_patch)
    broad_target = target.broadcast_var(conv_patch)
    #error
    squared_error = jt.sum((conv_patch * broad_mask - broad_target) ** 2, [2, 3, 4])
    min_error_location = np.argmin(squared_error)
    min_error_x, min_error_y = np.divmod(min_error_location, squared_error.shape[1])
    return min_error_x, min_error_y, squared_error[min_error_x, min_error_y]


def graph_cut(input_image, filling_image, mask,sinkcaps,sourcecaps):
    
    graph = maxflow.Graph[float]()
    idx = graph.add_grid_nodes(input_image.shape[:2])
    
    weight = np.abs(input_image - filling_image).sum(axis=2)
    #edge
    graph.add_grid_edges(idx, weight, structure=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), symmetric=True)
    
    #t-links
    source=np.float32(sourcecaps== 0) * 1e20
    sink=np.float32(sinkcaps == 0) * 1e20
    graph.add_grid_tedges(idx, sourcecaps=source, sinkcaps=sink)
    
    flow = graph.maxflow()
    print(f"Max flow: {flow}")
    
    segments = graph.get_grid_segments(idx)
    
    return segments

def build_poisson_matrix(mask):
    mask = mask.astype(bool)
    matrix_A = scipy.sparse.identity(np.prod(mask.shape[:2]), format='lil')
    for row in range(1, mask.shape[0] - 1):
        for column in range(1, mask.shape[1] - 1):
            if mask[row, column]:
                index = column + row * mask.shape[1]
                matrix_A[index, index] = 4
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for dx, dy in directions:
                    new_col, new_row = column + dx, row + dy
                    if 0 <= new_col < mask.shape[1] and 0 <= new_row < mask.shape[0]:
                        neighbor_idx = new_col + new_row * mask.shape[1]
                        matrix_A[index, neighbor_idx] = -1
    return matrix_A.tocsr()

def poisson_blend(base_img, blend_mask, overlay_img):
    matrix_A = build_poisson_matrix(blend_mask)
    poisson_grid = pyamg.gallery.poisson(blend_mask.shape)

    for color_channel in range(base_img.shape[2]):
        base_layer = base_img[:, :, color_channel].flatten()
        overlay_layer = overlay_img[:, :, color_channel].flatten()
        blend_vector = poisson_grid * overlay_layer
        for i in range(base_img.shape[0]):
            for j in range(base_img.shape[1]):
                if not blend_mask[i, j]:
                    idx = j + i * base_img.shape[1]
                    blend_vector[idx] = base_layer[idx]
        result, _ = scipy.sparse.linalg.cg(matrix_A, blend_vector, tol=1e-8)
        reshaped_result = np.reshape(result, (base_img.shape[0], base_img.shape[1]))
        reshaped_result = np.clip(reshaped_result, 0, 255)
        reshaped_result = reshaped_result.astype(base_img.dtype)
        base_img[:, :, color_channel] = reshaped_result

    return base_img


def main(original_image_path,mask_path,patch_image_path,output_directory):   
    mask = cv2.imread(mask_path, 0)
    original_image = cv2.imread(original_image_path)
    patch_image=cv2.imread(patch_image_path)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    binary_mask=255-binary_mask
    kernel_size = 60 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    #dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    kernel_radius_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask=cv2.dilate(binary_mask, kernel_radius_1, iterations=40)

    #sink and source are used in the graph cut process
    _, binary_mask_sink = cv2.threshold(dilated_mask, 127, 255, cv2.THRESH_BINARY)
    binary_mask_sink=255-binary_mask_sink
    edges_sink = cv2.Canny(binary_mask_sink, 100, 200)
    sink = cv2.bitwise_not(edges_sink)
    expanded_area = cv2.bitwise_xor(binary_mask, dilated_mask)

    region_of_interest = cv2.bitwise_and(original_image, original_image, mask=expanded_area)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges = cv2.Canny(binary_mask, 100, 200)
    source = cv2.bitwise_not(edges)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    mask_small = expanded_area[y:y+h, x:x+w]
    region_of_interest=region_of_interest[y:y+h, x:x+w]
    source=source[y:y+h, x:x+w]
    sink=sink[y:y+h, x:x+w]

    x1,y1,_=compute_error(region_of_interest,patch_image,mask_small)
    patch_image = np.asarray(patch_image)
    h1, w1 = region_of_interest.shape[1], region_of_interest.shape[0]
    patch_image = patch_image[x1:x1 + w1,y1:y1 + h1]
    segments=graph_cut(region_of_interest, patch_image, mask_small,sink,source)
    segments=segments.astype(np.uint8)

    #match patch, mask and original image 
    big_patch=np.zeros(original_image.shape)
    big_patch[y:y+h,x:x+w]=patch_image
    big_mask=np.zeros(original_image.shape[:2])
    big_mask[y:y+h,x:x+w]=1-segments
    output1 = original_image * (1 - np.expand_dims(big_mask, axis=2)) + big_patch * np.expand_dims(big_mask, axis=2)
    #start_time = time.time()
    output = poisson_blend(original_image, big_mask, big_patch)
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"Elapsed Time: {elapsed_time} seconds")
    big_mask = np.expand_dims(big_mask, axis=2)
    output = original_image * (1 - big_mask) + output * big_mask

    _, filename = os.path.split(original_image_path)
    name, extension = os.path.splitext(filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    new_name = name + '_result'
    new_image_path = os.path.join(output_directory, new_name + extension)
    cv2.imwrite(new_image_path, output)
    mid=name+'_nopossion'
    mid_p=os.path.join(output_directory, mid + extension)
    cv2.imwrite(mid_p, output1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('original_image_path', type=str,default='./data/completion/input2_mask.jpg', help='Path to the original image')
    parser.add_argument('mask_path', type=str,default='./data/completion/input2_mask.jpg', help='Path to the mask image')
    parser.add_argument('patch_image_path', type=str,default='./data/completion/input2_patch.jpg', help='Path to the patch image')
    parser.add_argument('output_directory', type=str, default='./result', help='Directory to save the result')
    args = parser.parse_args()
    main(args.original_image_path, args.mask_path, args.patch_image_path, args.output_directory)
