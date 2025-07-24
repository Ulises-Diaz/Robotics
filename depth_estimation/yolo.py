import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time

# Cargar modelo YOLO
model = YOLO('/home/uli/robotics/vision/depth_estimation/best.pt')

# Configuraciones
CONFIDENCE_THRESHOLD = 0.5

# Configurar pipeline RealSense
pipe = rs.pipeline()
cfg = rs.config()

# Habilitar streams de color y profundidad
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Iniciar pipeline
profile = pipe.start(cfg)

# Obtener la escala de profundidad
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()


# Alineador para sincronizar depth con color
align_to = rs.stream.color
align = rs.align(align_to)

# Variables para FPS
fps_counter = 0
start_time = time.time()

# Función para obtener distancia de las detecciones
def get_detection_distance(depth_frame, x1, y1, x2, y2):
    """
    Obtiene la distancia promedio del centro de la detección
    """
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # Obtener distancia del punto central
    distance = depth_frame.get_distance(center_x, center_y)
    return distance, center_x, center_y

try:

    show_depth = False
    
    
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
        
        # Hacer predicción YOLO en la imagen de color
        results = model(color_image, conf=CONFIDENCE_THRESHOLD)
        
        # Clonar imagen para anotar
        annotated_frame = color_image.copy()
        
        # Procesar detecciones y agregar información de profundidad
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # Obtener coordenadas y confianza
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Obtener distancia del centro de la detección
                distance, center_x, center_y = get_detection_distance(depth_frame, x1, y1, x2, y2)
                
                # Dibujar bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Dibujar punto central
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Preparar texto con clase, confianza y distancia
                class_name = model.names[cls]
                label = f'{class_name}: {conf:.2f}'
                distance_text = f'Dist: {distance:.2f}m'
                
                # Dibujar etiquetas
                cv2.putText(annotated_frame, label, (int(x1), int(y1-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(annotated_frame, distance_text, (int(x1), int(y1-30)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Calcular FPS
        fps_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = fps_counter / elapsed_time
            fps_counter = 0
            start_time = time.time()
            
            # Mostrar FPS
            cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar información adicional
        cv2.putText(annotated_frame, f'Conf: {CONFIDENCE_THRESHOLD:.2f}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated_frame, f'Detections: {num_detections}', (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Crear imagen combinada o mostrar solo RGB
        if show_depth:
            # Aplicar colormap al depth para visualización
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Combinar imágenes lado a lado
            combined_image = np.hstack((annotated_frame, depth_colormap))
            cv2.imshow('YOLO + RealSense (RGB + Depth)', combined_image)
        else:
            cv2.imshow('YOLO + RealSense (RGB)', annotated_frame)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()
    print("=== Programa terminado ===")
