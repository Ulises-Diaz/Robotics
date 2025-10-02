#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import yaml
from PIL import Image
import numpy as np
import os

class SimpleMapPublisher(Node):
    def __init__(self):
        super().__init__('simple_map_publisher')
        
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.timer = self.create_timer(1.0, self.publish_map)
        
        # Path al mapa
        home = os.path.expanduser('~')
        map_file = f'' #poner mapa
        
        if not os.path.exists(map_file):
            self.get_logger().error(f'No se encontró: {map_file}')
            return
        
        self.load_map(map_file)
        self.get_logger().info("✓ Mapa cargado y publicando en /map")
    
    def load_map(self, yaml_file):
        try:
            # Leer YAML
            with open(yaml_file, 'r') as f:
                map_config = yaml.safe_load(f)
            
            # Cargar imagen PGM
            map_dir = os.path.dirname(yaml_file)
            image_file = os.path.join(map_dir, map_config['image'])
            
            if not os.path.exists(image_file):
                self.get_logger().error(f'No se encontró la imagen: {image_file}')
                return
            
            img = Image.open(image_file).convert('L')
            img_array = np.array(img)
            
            # Convertir a occupancy grid (0-100)
            # Blanco (255) = libre (0), Negro (0) = ocupado (100)
            self.map_data = ((255 - img_array) / 255.0 * 100).astype(np.int8)
            
            self.resolution = map_config['resolution']
            self.origin = map_config['origin']
            self.height, self.width = self.map_data.shape
            
            self.get_logger().info(f"Dimensiones: {self.width}x{self.height}, res: {self.resolution}m")
        except Exception as e:
            self.get_logger().error(f'Error cargando mapa: {e}')
    
    def publish_map(self):
        if not hasattr(self, 'map_data'):
            return
            
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = "map"
        map_msg.header.stamp = self.get_clock().now().to_msg()
        
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.width
        map_msg.info.height = self.height
        map_msg.info.origin.position.x = float(self.origin[0])
        map_msg.info.origin.position.y = float(self.origin[1])
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        
        map_msg.data = self.map_data.flatten().tolist()
        
        self.map_pub.publish(map_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleMapPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()