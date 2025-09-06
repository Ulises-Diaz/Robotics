from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {} # team 1m team 2


    
    def get_clustering_model (self, image):
        # reshape image 2d array
        image_2d = image.reshape(-1,3)

        # perform kmeans with 2 clusters
        kmeans = KMeans(n_clusters= 2, init="k-means++", n_init=1).fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bounding_box): 
        image = frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Clustering model 

        kmeans = self.get_clustering_model(top_half_image)

        # get cluster labels for each pixel 
        labels = kmeans.labels_

        # reshape dim into image
        clustered_image = labels.reshape(int(top_half_image.shape[0]), int(top_half_image.shape[1]))

        # Get cluster
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1-non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color
    
    def assign_team_color(self, frame, player_detections):


        player_colors = []
        for _, player_detection in player_detections.items():
            bounding_box = player_detection["bounding_box"]
            player_color = self.get_player_color(frame, bounding_box)
            player_colors.append(player_color) # get all plahyer color within 1st frame

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init =1)
        kmeans.fit(player_colors) # get 2 colors

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bounding_box, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
    
        player_color = self.get_player_color(frame, player_bounding_box)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1


        self.player_team_dict[player_id] = team_id

        return team_id
