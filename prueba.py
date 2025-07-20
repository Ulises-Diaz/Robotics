import pyrealsense2 as rs
import numpy as np
import cv2

# Configurar pipeline
pipe = rs.pipeline()
cfg = rs.config()

# Habilitar streams de color Y profundidad
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Iniciar pipeline
profile = pipe.start(cfg)

# Obtener la escala de profundidad (conversión a metros)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Escala de profundidad: {depth_scale}")

# Alineador para sincronizar depth con color
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # Esperar frames
        frames = pipe.wait_for_frames()
        
        # Alinear depth frame con color frame
        aligned_frames = align.process(frames)
        
        # Obtener frames alineados
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
            
        # Convertir a arrays numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Aplicar colormap al depth para visualización
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # Función para obtener distancia al hacer clic
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Obtener valor de profundidad en píxel clickeado
                depth_value = depth_frame.get_distance(x, y)
                print(f"Coordenadas: ({x}, {y}) - Distancia: {depth_value:.3f} metros")
                
                # # Dibujar punto en la imagen
                # cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)
                # cv2.putText(color_image, f"{depth_value:.2f}m", 
                #            (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                #            0.5, (0, 0, 255), 2)
        
        # Crear imagen combinada
        images = np.hstack((color_image, depth_colormap))
        
        # Mostrar instrucciones
        # cv2.putText(images, "Haz clic para medir distancia", 
        #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # cv2.putText(images, "Presiona 'q' para salir", 
        #            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar imagen
        cv2.imshow('RGB + Distancia', images)
        
        # Configurar callback del mouse (después de crear la ventana)
        cv2.setMouseCallback('RGB + Distancia', mouse_callback)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    pipe.stop()
    cv2.destroyAllWindows()