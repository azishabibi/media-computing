import cv2
import numpy as np
import maxflow

# Load the images and the overlap mask
input1_path = 'fillin_image.png'
fill_image_path = 'region_image.png'
overlap_mask_path = 'output_image.png'

input1_img = cv2.imread(input1_path)
fill_img = cv2.imread(fill_image_path)
overlap_mask_img = cv2.imread(overlap_mask_path, cv2.IMREAD_GRAYSCALE)

# Make sure that the fill image is the same size as the input image
fill_img = cv2.resize(fill_img, (input1_img.shape[1], input1_img.shape[0]), interpolation=cv2.INTER_LINEAR)

# Create boolean mask where the white area is True and the rest is False
overlap_mask = overlap_mask_img > 128

def graph_cut_stitch(input1, fill, mask):
    """
    Apply graph-cut algorithm to stitch two images using the mask.
    """
    # Create a graph
    g = maxflow.GraphFloat()

    # Add nodes
    nodeids = g.add_grid_nodes(input1.shape[:2])

    # Add edges with weights based on the color differences in the overlap region
    # Weights are the sum of absolute differences across color channels
    structure = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    weights = np.abs(input1.astype(np.int32) - fill.astype(np.int32)).sum(axis=2) * mask
    g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=True)

    structure = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=True)

    # Set terminal edges
    g.add_grid_tedges(nodeids, mask.astype(np.float32) * 1e9, (1 - mask).astype(np.float32) * 1e9)

    # Find the maximum flow
    g.maxflow()

    # Get the segments of the nodes in the grid
    sgm = g.get_grid_segments(nodeids)

    # Create a mask from the segments
    mask = np.int_(np.logical_not(sgm))

    # Use the mask to blend the images
    result_img = input1 * mask[:, :, np.newaxis] + fill * (1 - mask)[:, :, np.newaxis]

    return result_img

# Perform graph cut stitching
result = graph_cut_stitch( fill_img,input1_img, overlap_mask)

# Save the result
result_path = 'result.jpg'
cv2.imwrite(result_path, result)
result_path
