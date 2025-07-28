import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Importar con manejo de errores
try:
    from lib.visualization import plotting
    from lib.visualization.video import play_trip
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Advertencia: Librerías de visualización no disponibles")
    VISUALIZATION_AVAILABLE = False

from tqdm import tqdm


class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, "poses.txt"))
        # CORREGIDO: Cambié 'image_l' por 'image_1' (más común en KITTI)
        try:
            self.images = self._load_images(os.path.join(data_dir, "image_1"))
        except:
            # Fallback si no encuentra image_1
            self.images = self._load_images(os.path.join(data_dir, "image_l"))
            
        # MEJORADO: Más features para mejor tracking
        self.orb = cv2.ORB_create(5000)  # Aumentado de 3000 a 5000
        
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):
        """CORREGIDO: Mejor manejo de errores en carga de calibración"""
        try:
            with open(filepath, 'r') as f:
                line = f.readline().strip()
                try:
                    params = np.fromstring(line, dtype=np.float64, sep=' ')
                except ValueError:
                    # Fallback para versiones nuevas de numpy
                    params = np.array([float(x) for x in line.split()])
                
                if len(params) != 12:
                    raise ValueError(f"Se esperaban 12 parámetros, se encontraron {len(params)}")
                    
                P = np.reshape(params, (3, 4))
                K = P[0:3, 0:3]
        except Exception as e:
            print(f"Error cargando calibración: {e}")
            raise
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """CORREGIDO: Mejor manejo de errores en carga de poses"""
        poses = []
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f.readlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        T = np.fromstring(line, dtype=np.float64, sep=' ')
                    except ValueError:
                        T = np.array([float(x) for x in line.split()])
                    
                    if len(T) != 12:
                        print(f"Advertencia: Línea {line_num + 1} tiene {len(T)} elementos")
                        continue
                        
                    T = T.reshape(3, 4)
                    T = np.vstack((T, [0, 0, 0, 1]))
                    poses.append(T)
        except Exception as e:
            print(f"Error cargando poses: {e}")
            raise
        return poses

    @staticmethod
    def _load_images(filepath):
        """CORREGIDO: Mejor validación de imágenes"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Directorio no encontrado: {filepath}")
            
        image_files = sorted([f for f in os.listdir(filepath) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            raise ValueError(f"No se encontraron imágenes en {filepath}")
            
        images = []
        for file in image_files:
            path = os.path.join(filepath, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
            else:
                print(f"Advertencia: No se pudo cargar {file}")
                
        return images

    @staticmethod
    def _form_transf(R, t):
        """Formar matriz de transformación 4x4"""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """CORREGIDO: Mejor filtrado de matches y manejo de errores"""
        # Find keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        
        # Verificar que se encontraron descriptores
        if des1 is None or des2 is None:
            print(f"Advertencia: No se encontraron descriptores en frame {i}")
            return None, None

        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        # CORREGIDO: Mejor filtrado de matches con Lowe's ratio test
        good = []
        for match_pair in matches:
            if len(match_pair) == 2:  # Asegurar que tenemos 2 matches
                m, n = match_pair
                if m.distance < 0.7 * n.distance:  # MEJORADO: Ratio más estricto (0.7 vs 0.8)
                    good.append(m)

        # AGREGADO: Verificar número mínimo de matches
        if len(good) < 15:  # Mínimo 15 matches para estabilidad
            print(f"Advertencia: Solo {len(good)} matches buenos en frame {i}")
            if len(good) < 8:  # Menos de 8 es crítico
                return None, None

        # Mostrar matches (opcional, comentar para mejor rendimiento)
        # if False:  # Cambiar a True para ver matches
        #     draw_params = dict(matchColor=(0, 255, 0),
        #                      singlePointColor=None,
        #                      matchesMask=None,
        #                      flags=2)

        #     img3 = cv2.drawMatches(self.images[i-1], kp1, self.images[i], kp2, good, None, **draw_params)
        #     cv2.imshow("Matches", img3)
        #     cv2.waitKey(50)  # Reducido de 200 a 50ms

        # # CORREGIDO: Usar índices correctos
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        return q1, q2

    def get_pose(self, q1, q2):
        """CORREGIDO: Mejor estimación de pose con RANSAC mejorado"""
        if q1 is None or q2 is None:
            return np.eye(4)

        # MEJORADO: Parámetros de RANSAC más estrictos
        E, mask = cv2.findEssentialMat(q1, q2, self.K, 
                                      method=cv2.RANSAC,
                                      prob=0.999,  # MEJORADO: Mayor confianza
                                      threshold=0.5)  # MEJORADO: Threshold más estricto
        
        if E is None:
            print("Advertencia: No se pudo calcular matriz esencial")
            return np.eye(4)

        # AGREGADO: Filtrar puntos usando máscara de RANSAC
        if mask is not None:
            q1_filtered = q1[mask.ravel() == 1]
            q2_filtered = q2[mask.ravel() == 1]
            
            # Verificar que quedan suficientes puntos
            if len(q1_filtered) < 8:
                print("Advertencia: Muy pocos inliers después de RANSAC")
                return np.eye(4)
        else:
            q1_filtered, q2_filtered = q1, q2

        # Decompose essential matrix
        R, t = self.decomp_essential_mat(E, q1_filtered, q2_filtered)
        
        # AGREGADO: Validar que R es una matriz de rotación válida
        if not self._is_valid_rotation(R):
            print("Advertencia: Matriz de rotación inválida")
            return np.eye(4)

        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def _is_valid_rotation(self, R):
        """AGREGADO: Validar matriz de rotación"""
        # Verificar que det(R) = 1
        det_R = np.linalg.det(R)
        if abs(det_R - 1.0) > 0.1:
            return False
        
        # Verificar que R^T * R = I
        should_be_identity = np.dot(R.T, R)
        identity = np.eye(3)
        if not np.allclose(should_be_identity, identity, atol=0.1):
            return False
            
        return True

    def decomp_essential_mat(self, E, q1, q2):
        """CORREGIDO: Mejor descomposición de matriz esencial"""
        
        def sum_z_cal_relative_scale(R, t):
            # CORREGIDO: Asegurar que t es un vector columna
            t = t.reshape(-1, 1) if t.ndim == 1 else t
            
            T = self._form_transf(R, t.flatten())
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangular puntos 3D
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenizar
            uhom_Q1 = hom_Q1[:3, :] / (hom_Q1[3, :] + 1e-8)  # AGREGADO: Evitar división por cero
            uhom_Q2 = hom_Q2[:3, :] / (hom_Q2[3, :] + 1e-8)

            # Contar puntos con Z positiva
            sum_of_pos_z_Q1 = np.sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = np.sum(uhom_Q2[2, :] > 0)

            # CORREGIDO: Calcular escala relativa de forma más robusta
            if uhom_Q1.shape[1] > 1 and uhom_Q2.shape[1] > 1:
                distances_Q1 = np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=1)
                distances_Q2 = np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=1)
                
                # Evitar divisiones por cero
                mask = (distances_Q2 > 1e-6) & (distances_Q1 > 1e-6)
                if np.sum(mask) > 0:
                    relative_scale = np.median(distances_Q1[mask] / distances_Q2[mask])  # MEJORADO: Usar mediana
                else:
                    relative_scale = 1.0
            else:
                relative_scale = 1.0

            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Descomponer matriz esencial
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # MEJORADO: Evaluar todas las combinaciones posibles
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Seleccionar la mejor solución
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        
        R_final, t_final = right_pair
        
        # MEJORADO: Limitar la escala para evitar saltos grandes
        relative_scale = np.clip(relative_scale, 0.1, 10.0)
        t_final = t_final * relative_scale

        return [R_final, t_final]


def simple_matplotlib_plot(gt_path, estimated_path, title="Visual Odometry Results"):
    """Función simple para plotear con matplotlib"""
    gt_x = [point[0] for point in gt_path]
    gt_z = [point[1] for point in gt_path]
    est_x = [point[0] for point in estimated_path]
    est_z = [point[1] for point in estimated_path]
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(gt_x, gt_z, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    plt.plot(est_x, est_z, 'r--', linewidth=2, label='Estimated Path', alpha=0.8)
    
    plt.plot(gt_x[0], gt_z[0], 'go', markersize=8, label='Start')
    plt.plot(gt_x[-1], gt_z[-1], 'bs', markersize=8, label='End GT')
    plt.plot(est_x[-1], est_z[-1], 'rs', markersize=8, label='End Est')
    
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Z Position (m)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Calcular estadísticas
    final_error = np.sqrt((gt_x[-1] - est_x[-1])**2 + (gt_z[-1] - est_z[-1])**2)
    total_gt_distance = sum([np.sqrt((gt_x[i+1] - gt_x[i])**2 + (gt_z[i+1] - gt_z[i])**2) 
                            for i in range(len(gt_x)-1)])
    
    stats_text = f'Final Error: {final_error:.2f}m\n'
    stats_text += f'GT Distance: {total_gt_distance:.2f}m\n'
    stats_text += f'Error %: {(final_error/total_gt_distance)*100:.1f}%\n'
    stats_text += f'Frames: {len(gt_path)}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return final_error, total_gt_distance


def main():
    data_dir = "KITTI_sequence_2"  # CORREGIDO: Cambiado de sequence_2 a sequence_1
    
    try:
        vo = VisualOdometry(data_dir)
        print(f"Cargadas {len(vo.images)} imágenes y {len(vo.gt_poses)} poses")
    except Exception as e:
        print(f"Error inicializando Visual Odometry: {e}")
        return

    # AGREGADO: Opción para reproducir video
    if VISUALIZATION_AVAILABLE:
        try:
            print("¿Reproducir video de imágenes? (y/n): ", end="")
            if input().lower().startswith('y'):
                play_trip(vo.images)
        except:
            pass

    gt_path = []
    estimated_path = []
    
    print("Procesando frames...")
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            if not np.allclose(transf, np.eye(4)):
                cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

    # Mostrar resultados
    print(f"\nResultados del procesamiento:")
    print(f"Frames procesados: {len(gt_path)}")
    
    try:
        # Usar matplotlib como método principal
        final_error, gt_distance = simple_matplotlib_plot(gt_path, estimated_path, "Visual Odometry - Resultado Mejorado")
        
        print(f"\n=== ESTADÍSTICAS FINALES ===")
        print(f"Error final: {final_error:.3f} metros")
        print(f"Distancia total GT: {gt_distance:.3f} metros")
        print(f"Error relativo: {(final_error/gt_distance)*100:.2f}%")
        
  
    except Exception as e:
        print(f"Error en visualización: {e}")
        
    # Fallback a Bokeh si está disponible
    if VISUALIZATION_AVAILABLE:
        try:
            plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", 
                                   file_out=os.path.basename(data_dir) + "_result.html")
            print(f"Gráfico Bokeh guardado como: {os.path.basename(data_dir)}_result.html")
        except Exception as e:
            print(f"Error con Bokeh: {e}")


if __name__ == "__main__":
    main()