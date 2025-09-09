import sys
sys.path.append('../')
from utils import get_center_bounding_box, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_box):
        ball_position = get_center_bounding_box(ball_box)

        minimum_distance = 9999
        assigned_player = -1 

        for player_id , player in players.items(): 
            player_bounding_box = player["bounding_box"]

            distance_left_foot = measure_distance((player_bounding_box[0], player_bounding_box[-1]),ball_position)
            distance_right_foot = measure_distance((player_bounding_box[2], player_bounding_box[-1]),ball_position)
            distance = min(distance_left_foot, distance_right_foot)

            if distance < self.max_player_ball_distance: 
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player




        