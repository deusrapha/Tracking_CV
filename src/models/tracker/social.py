import numpy as np

def calculate_herd_velocity(tracks):
    """
    Computes the average velocity vector of visible cows to serve as the social herd prior.
    """
    visible_velocities = []
    for track in tracks:
        if track.state in ["VISIBLE", "NEW"]:
            v = track.motion["x"][2, 0]
            theta = track.motion["x"][3, 0]
            vx = v * np.cos(theta)
            vy = v * np.sin(theta)
            visible_velocities.append([vx, vy])
            
    if len(visible_velocities) > 0:
        mean_vx = sum(v[0] for v in visible_velocities) / len(visible_velocities)
        mean_vy = sum(v[1] for v in visible_velocities) / len(visible_velocities)
        return [float(mean_vx), float(mean_vy)]
    return None
