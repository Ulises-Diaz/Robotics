from PIL import Image
import numpy as np

map_array = np.ones((400, 400), dtype=np.uint8) * 255 

# # Añade algunos obstáculos (negro)
# map_array[100:120, 50:350] = 0  # Pared horizontal
# map_array[50:350, 100:120] = 0  # Pared vertical

img = Image.fromarray(map_array, mode='L')
img.save('test_map.pgm')
print("Map created: test_map.pgm")