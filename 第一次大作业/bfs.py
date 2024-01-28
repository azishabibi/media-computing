from PIL import Image
import numpy as np

# Load the mask image
mask_path = './data/completion/input1_mask.jpg'
mask_img = Image.open(mask_path)
mask = np.array(mask_img.convert('1'))

# Define the BFS function to find the boundary within K pixels
def bfs(mask, k):
    #import pdb;pdb.set_trace()
    #mask=mask.convert('1')
    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 4-connected neighbours
    queue = []

    # Initialize the queue with the boundary pixels
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] == 0:
                for dr, dc in directions:
                    ni, nj = i + dr, j + dc
                    if 0 <= ni < rows and 0 <= nj < cols and mask[ni, nj] == 255 and not visited[ni, nj]:
                        visited[ni, nj] = True
                        queue.append((ni, nj))

    # Perform BFS to a depth of k pixels
    for _ in range(k):
        next_queue = []
        while queue:
            i, j = queue.pop(0)
            for dr, dc in directions:
                ni, nj = i + dr, j + dc
                if 0 <= ni < rows and 0 <= nj < cols and not visited[ni, nj]:
                    visited[ni, nj] = True
                    next_queue.append((ni, nj))
        queue = next_queue

    # Generate the resulting mask for region B
    region_b = np.where(visited, 255, 0).astype(np.uint8)
    return region_b

# Get the region within K pixels of the missing area
k = 80
region_b = bfs(mask, k)

# Save the result to an image file
result_path = 'region_b.jpg'
Image.fromarray(region_b).save(result_path)

