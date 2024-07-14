# Inside the draw_boxes function or another suitable place

# Function to check collision based on speed and distance
def check_collision(speed1, speed2, distance):
    # Define thresholds
    distance_threshold = 50  # Example: 50 pixels
    speed_threshold = 10  # Example: 10 km/h

    # Check conditions for collision warning
    if distance < distance_threshold and abs(speed1 - speed2) < speed_threshold:
        return True
    else:
        return False

# Loop through each pair of tracked vehicles
for id1, id2 in all_possible_pairs_of_ids:
    # Get speeds and distances
    speed1 = get_speed_of_vehicle(id1)
    speed2 = get_speed_of_vehicle(id2)
    distance = calculate_distance_between_vehicles(id1, id2)
    
    # Check for collision
    if check_collision(speed1, speed2, distance):
        # Trigger collision warning
        print(f"Collision warning: Vehicles {id1} and {id2} are too close!")
        # Optionally, add visual or audio feedback

