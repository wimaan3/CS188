import cv2
import numpy as np
# or slam2d.png or slam3d.png 
raw = cv2.imread('images/slam2d-2.png', cv2.IMREAD_GRAYSCALE)
_, occ = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite('pgm/map2d-2.pgm', occ)

# or map.pgm or map2d-2.pgm or map3d-2.pgm or map3d.pgm
yaml_content = f"""image: pgm/map2d-2.pgm
resolution: 0.05
origin: [0.0, 0.0, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
with open('yaml/map2d-2.yaml', 'w') as f:
    f.write(yaml_content)
# or map.yaml or map3d-2.yaml
print("map3d.pgm and map3d.yaml created from slam3d.png")
