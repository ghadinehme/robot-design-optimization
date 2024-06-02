
from setup import *

def display(best_individual, reachable_points, wall, base, origin, basis, shapes, limits, points = True, ani = False):
    r = 0.1
    vars = best_individual
    links = []
    axis = ["x", "y", "z"]
    n = 0
    n_flip = 0
    lengths = 0
    while len(vars) > 4:
        if vars[0] == 0:
            var = vars[:4]
            length = var[2]
            links.append(RevoluteJoint(axis[var[1]], length, r, axis[var[3]]))
            lengths += length
            n+=1
            vars = vars[4:]
        elif vars[0] == 1:
            var = vars[:4]
            length = var[2]
            dmax = var[3]
            links.append(PrismaticJoint(axis[var[1]], length, r, dmax))
            lengths += length
            lengths += dmax
            n+=1
            vars = vars[4:]
        else:
            var = vars[:5]
            length = var[4]
            links.append(Flip(axis[var[1]], var[3], length, r, axis[var[2]]))
            lengths += length**2
            n_flip+=1
            vars = vars[5:]
    thetas = [0 for i in range(n)]
    robot = Robot(base, origin, basis, links, thetas)
    print("Length of the robot: ", lengths)
    j = ""
    for link in links:
        if link.type == "revolute":
            j += "R"
        elif link.type == "prismatic":
            j += "P"
        else:
            j += "F"
    print(j)
    if points:
        robot.points_plot(reachable_points, wall, shapes, limits, ani)
    else:
        robot.interactive_plot()

def generate_trajectory(time_steps=50):
    """
    Generate a complex trajectory for a dental robot end effector.

    Parameters:
    time_steps (int): Number of time steps in the trajectory.

    Returns:
    np.ndarray: Array of shape (time_steps, 6) representing the trajectory.
                Each row is [x, y, z, roll, pitch, yaw].
    """
    t = np.linspace(0, 2*np.pi, time_steps)

    # Define the trajectory components
    x = 3 * np.sin(2 * t)  # Sinusoidal motion in x-direction
    y = 3 * np.cos(3 * t)  # Sinusoidal motion in y-direction
    z = 2 * np.sin(4 * t) # Sinusoidal motion in z-direction

    return x, y, z

def generate_trajectory(trajectory_type='sinusoidal', ax = 1, ay = 1, az = 1, fx = 2, fy = 2, fz = 2, time_steps=50):
    """
    Generate a complex trajectory for a dental robot end effector.

    Parameters:
    time_steps (int): Number of time steps in the trajectory.
    trajectory_type (str): Type of trajectory ('sinusoidal', 'spiral', 'random').

    Returns:
    np.ndarray: Array of shape (time_steps, 6) representing the trajectory.
                Each row is [x, y, z, roll, pitch, yaw].
    """
    t = np.linspace(0, 2*np.pi, time_steps)

    if trajectory_type == 'sinusoidal':
        # Sinusoidal motion
        x = ax * np.sin(fx * t)
        y = ay * np.cos(fy * t)
        z = az * np.sin(fz * t)
        
    elif trajectory_type == 'spiral':
        # Spiral motion
        x = ax * t * np.sin(fx * t)
        y = ay * t * np.cos(fy * t)
        z = az * t
        
    elif trajectory_type == 'random':
        # Random motion
        x = ax * (np.random.rand(time_steps) - 0.5)
        y = ay * (np.random.rand(time_steps) - 0.5)
        z = az * (np.random.rand(time_steps) - 0.5)
        
    else:
        raise ValueError("Unsupported trajectory type. Choose 'sinusoidal', 'spiral', or 'random'.")


    return x, y, z


def generate_complex_closed_loop(n_points=100):
    # Generate angles for parameterization
    t = np.linspace(0, 2 * np.pi, n_points)
    
    # Parametric equations for a complex 3D loop
    x = np.sin(t) + np.sin(3 * t) * 0.3
    y = np.cos(t) + np.cos(3 * t) * 0.3
    z = np.sin(5 * t) * 0.2
    
    return x, y, z

def generate_curved_plane(x_min, x_max, y_min, y_max, z_min, z_max, n_points):
    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) + np.cos(Y)
    Z = Z * (z_max - z_min) + z_min
    return X, Y, Z


def generate_points_on_sphere_with_hole(num_points, hole_radius, hole_direction, sphere_radius=1.0):
    """
    Generate points on a sphere with a hole of radius `hole_radius` in the direction `hole_direction`.
    
    Parameters:
    - num_points: Total number of points to generate on the sphere.
    - hole_radius: Radius of the hole on the sphere.
    - hole_direction: Direction of the hole on the sphere (a 3D unit vector).
    - sphere_radius: Radius of the sphere. Default is 1.0.
    
    Returns:
    - points: An array of points on the sphere excluding the hole region.
    """
    def normalize(v):
        return v / np.linalg.norm(v)
    
    hole_direction = normalize(hole_direction)
    hole_cos_angle = np.cos(hole_radius / sphere_radius)
    
    points = []
    while len(points) < num_points:
        # Generate a random point on the sphere
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)
        
        x = sphere_radius * np.sin(theta) * np.cos(phi)
        y = sphere_radius * np.sin(theta) * np.sin(phi)
        z = sphere_radius * np.cos(theta)
        
        point = np.array([x, y, z])
        
        # Check if the point is within the hole region
        cos_angle = np.dot(normalize(point), hole_direction)
        if cos_angle < hole_cos_angle:
            points.append(point)
    
    return np.array(points)