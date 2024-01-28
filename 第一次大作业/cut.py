import numpy as np
import cv2
import networkx as nx
from scipy.spatial.distance import euclidean

def build_graph(image, mask):
    rows, cols = image.shape[:2]
    graph = nx.Graph()
    node_id_map = {}  # 添加这一行
    graph.add_node(0, source_capacity=0, sink_capacity=1)
    graph.add_node(1, source_capacity=1, sink_capacity=0)
    for r in range(rows):
        for c in range(cols):
            node_id = r * cols + c
            node_id_map[node_id] = (r,c)

            if mask[r, c] == 255:  # Foreground
                graph.add_node(node_id, source_capacity=0, sink_capacity=1)
            else:  # Background
                graph.add_node(node_id, source_capacity=1, sink_capacity=0)

            # Connect neighboring pixels
            if c < cols - 1:
                weight = euclidean(image[r, c], image[r, c + 1])
                graph.add_edge(node_id, node_id + 1, weight=weight)

            if r < rows - 1:
                weight = euclidean(image[r, c], image[r + 1, c])
                graph.add_edge(node_id, node_id + cols, weight=weight)

    return graph, node_id_map  # 修改这一行

def graph_cut(image, mask):
    graph, node_id_map = build_graph(image, mask)  # 修改这一行
    edges_with_weights = graph.edges(data=True)

    for edge in edges_with_weights:
        u, v, data = edge
        weight = data.get('weight', None)
        
        if weight is not None and weight < 0:
            print(f"Edge ({u}, {v}) has negative weight: {weight}")
    is_connected = nx.is_connected(graph)
    print("Is the graph connected?", is_connected)
    min_cut, partition = nx.minimum_cut(graph, 0, 1,flow_func=nx.algorithms.flow.boykov_kolmogorov)

    foreground_nodes, _ = partition

    rows, cols = image.shape[:2]
    result_mask = np.zeros((rows, cols), dtype=np.uint8)
    import pdb;pdb.set_trace()
    for node_id in foreground_nodes:
        #print("Foreground Nodes:", list(foreground_nodes)[0:10])
        #print("Node ID Map Keys:", list(node_id_map.keys())[0:10])
        try:
            r, c = node_id_map[node_id]
        except KeyError as e:
            print(f"KeyError: {e}, node_id: {node_id}, node_id_map: {node_id_map}")
            continue
        result_mask[r, c] = 255

    return result_mask


def apply_blend(image_A, image_B, mask):
    result = np.copy(image_B)
    result[mask == 255] = image_A[mask == 255]
    return result


# Load your blurred image
blurred_image = cv2.imread('extracted_region.jpg')

# Create a mask for the region to be segmented
# For simplicity, you can manually create a mask or use a segmentation algorithm
# In this example, I assume you have a binary mask where the region to be segmented is set to 1
mask = cv2.imread('output_image.png', cv2.IMREAD_GRAYSCALE)
#import pdb;pdb.set_trace()
# Perform graph cut segmentation
segmentation_mask = graph_cut(blurred_image, mask)
import pdb;pdb.set_trace()
# Load your candidate image (image A)
image_A = cv2.imread('1.jpg')

# Apply the blend using the segmentation mask
result_image = apply_blend(image_A, blurred_image, segmentation_mask)
#import pdb;pdb.set_trace()
# Display the result
cv2.imwrite('Result_Image.jpg', result_image)

