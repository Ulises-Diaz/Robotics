#!/usr/bin/env python3

import rclpy 
from rclpy.node import Node 
import numpy as np 
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import heapq 

class Astar(Node): 
    def __init__(self): 
        super().__init__('Aestrella') 

        # Parameters 
        self.map_data = None 
        self.map_resolution = 0.05
        self.map_origin = None 
        self.map_width = 0 
        self.map_height = 0 

        # Subscriber
        self.map_sub = self.create_subscription(
            OccupancyGrid, 
            '/map',
            self.map_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped, 
            '/goal_pose',
            self.goal_callback,
            10
        )

        # Publisher
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoint_markers', 10)

        # Waypoints
        self.start_point = None
        self.end_point = None
        self.waypoints = []
        
        self.get_logger().info("A* Planner initialized")
        self.get_logger().info("Paso 1: Coloca el punto de INICIO con '2D Goal Pose'")

    def map_callback(self, msg): 
        """Occupancy Grid Map"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution 
        self.map_origin = msg.info.origin
        self.map_width = msg.info.width 
        self.map_height = msg.info.height
        self.get_logger().info(f"Mapa recibido: {self.map_width}x{self.map_height}")

    def goal_callback(self, msg): 
        """Handle new goal - Espera inicio y fin, luego genera waypoints"""
        if self.start_point is None:
            # Primer click = punto de inicio
            self.start_point = msg
            self.get_logger().info(f"Punto de INICIO: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
            self.get_logger().info("Paso 2: Coloca el punto FINAL con '2D Goal Pose'")
        elif self.end_point is None:
            # Segundo click = punto final
            self.end_point = msg
            self.get_logger().info(f"Punto FINAL: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
            self.generate_staircase_waypoints()
        else:
            # Si ya hay inicio y fin, reiniciar
            self.get_logger().info("Reiniciando. Coloca nuevo punto de INICIO")
            self.start_point = msg
            self.end_point = None
            self.waypoints = []

    def generate_staircase_waypoints(self):
        """Genera 5 waypoints en escalera entre inicio y fin"""
        if self.start_point is None or self.end_point is None:
            return
        
        start_x = self.start_point.pose.position.x
        start_y = self.start_point.pose.position.y
        end_x = self.end_point.pose.position.x
        end_y = self.end_point.pose.position.y
        
        # Generar 5 waypoints (inicio + 3 intermedios + fin = 5 total)
        self.waypoints = []
        num_waypoints = 5
        
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)  # Interpolación de 0 a 1
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)
            
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            self.waypoints.append(pose)
        
        self.get_logger().info(f"Generados {len(self.waypoints)} waypoints en escalera")
        self.visualize_waypoints()
        
        # Planificar ruta
        if len(self.waypoints) >= 2:
            self.plan_through_waypoints()

    def world_to_map(self, x, y): 
        """Convert world to map coordinates"""
        mx = int((x - self.map_origin.position.x) / self.map_resolution)
        my = int((y - self.map_origin.position.y) / self.map_resolution)
        return mx, my  

    def map_to_world(self, mx, my):
        """Convert map to world coordinates"""
        x = mx * self.map_resolution + self.map_origin.position.x
        y = my * self.map_resolution + self.map_origin.position.y
        return x, y

    def is_valid(self, x, y):
        """Check if a cell is valid and not occupied"""
        if x < 0 or x >= self.map_width or y < 0 or y >= self.map_height:
            return False
        if self.map_data[y, x] > 50:
            return False
        if self.map_data[y, x] < 0:
            return False
        return True
    
    def heuristic(self, a, b):
        """Euclidean distance heuristic"""
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    def get_neighbors(self, node):
        """Get valid neighbors (8-connected grid)"""
        x, y = node
        neighbors = []
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                neighbors.append(((nx, ny), cost))
        return neighbors
    
    def astar(self, start, goal):
        """A* pathfinding algorithm"""
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
            
            for next_node, move_cost in self.get_neighbors(current):
                new_cost = cost_so_far[current] + move_cost
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(goal, next_node)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
        
        # Reconstruct path
        if goal not in came_from:
            self.get_logger().warn("No se encontró ruta!")
            return []
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path
    
    def plan_through_waypoints(self):
        """Plan a path through all waypoints"""
        if self.map_data is None:
            self.get_logger().warn("No se ha recibido el mapa aún!")
            return
        
        full_path = []
        
        for i in range(len(self.waypoints) - 1):
            start_pose = self.waypoints[i]
            goal_pose = self.waypoints[i + 1]
            
            start_x, start_y = self.world_to_map(
                start_pose.pose.position.x,
                start_pose.pose.position.y
            )
            goal_x, goal_y = self.world_to_map(
                goal_pose.pose.position.x,
                goal_pose.pose.position.y
            )
            
            self.get_logger().info(f"Planificando segmento {i} -> {i+1}")
            
            segment_path = self.astar((start_x, start_y), (goal_x, goal_y))
            
            if not segment_path:
                self.get_logger().warn(f"Falló la búsqueda de ruta para segmento {i} -> {i+1}")
                return
            
            full_path.extend(segment_path)
        
        self.publish_path(full_path)
        self.get_logger().info(f"Ruta publicada con {len(full_path)} puntos")
    
    def publish_path(self, path):
        """Publish the path as a nav_msgs/Path"""
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for (mx, my) in path:
            x, y = self.map_to_world(mx, my)
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
    
    def visualize_waypoints(self):
        """Visualize waypoints as markers in RViz2"""
        marker_array = MarkerArray()
        
        for i, waypoint in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = waypoint.pose
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)
            
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "waypoint_labels"
            text_marker.id = i + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose = waypoint.pose
            text_marker.pose.position.z = 0.5
            text_marker.scale.z = 0.3
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"WP{i}"
            marker_array.markers.append(text_marker)
        
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    planner = Astar()
    
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()