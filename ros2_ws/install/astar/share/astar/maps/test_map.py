from PIL import Image
import numpy as np

# Crea un mapa 400x400 (20m x 20m a 0.05m/pixel)
map_array = np.ones((400, 400), dtype=np.uint8) * 255  # Todo libre (blanco)

# Añade algunos obstáculos (negro)
map_array[100:120, 50:350] = 0  # Pared horizontal
map_array[50:350, 100:120] = 0  # Pared vertical

# Guarda como PGM
img = Image.fromarray(map_array, mode='L')
img.save('test_map.pgm')
print("Mapa creado: test_map.pgm")