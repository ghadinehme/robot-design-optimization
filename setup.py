import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def transformation_matrix(theta, a, d, alpha, axis1 = "x", axis2 = "y"):
    """Transformation matrix given the parameters"""
    def rotation(axis, angle):
        d = {"x" : np.array([[1,0,0,0],
                             [0,np.cos(angle), -np.sin(angle), 0],
                             [0, np.sin(angle), np.cos(angle), 0],
                             [0,0,0,1]]),
             "y" : np.array([[np.cos(angle),0,np.sin(angle),0],
                            [0,1,0, 0],
                            [-np.sin(angle), 0, np.cos(angle), 0],
                            [0,0,0,1]]),
             "z" : np.array([[np.cos(angle),-np.sin(angle),0,0],
                            [np.sin(angle), np.cos(angle),0,0],
                            [0,0,1,0],
                            [0,0,0,1]])}
        return d[axis]
    
    def translation(axis, l):
        d2 = {"x" : 0, "y" : 1, "z" : 2}
        D = np.eye(4)
        D[d2[axis], 3] = l
        return D
    
    T = rotation(axis1, alpha) @ translation(axis1, a) @ rotation(axis2, theta) @ translation(axis2, d)
    
    return T

class Cylinder:
    def __init__(self, center, direction, radius, height):
        self.center = center
        self.direction = direction/np.linalg.norm(direction)
        self.radius = radius
        self.height = height

    def volume(self):
        return np.pi * self.radius**2 * self.height

    def surface_area(self):
        return 2 * np.pi * self.radius * (self.radius + self.height)
    
    def bottom_center(self):
        return self.center - self.direction * self.height / 2
    
    def top_center(self):
        return self.center + self.direction * self.height / 2
    
    def check_point_inside(self, point):
        bottom = self.bottom_center()
        top = self.top_center()
        direction = self.direction
        radius = self.radius
        height = self.height
        center = self.center
        
        # Check if the point is inside the cylinder
        v = point - top
        proj = np.dot(v, direction)
        if abs(proj) > height/2:
            return False
        dist = np.linalg.norm(v - proj * direction)
        return dist <= radius
    
    def grid_on_surface(self, n=100):
        # Normalize direction vector
        direction = self.direction
        radius = self.radius
        length = self.height
        center = self.center
        
        # Create rotation matrix to align the z-axis with the direction vector
        z_axis = np.array([0, 0, 1])
        if np.allclose(direction, z_axis):
            rotation = np.eye(3)
        else:
            axis = np.cross(z_axis, direction)
            theta = np.arccos(np.dot(z_axis, direction))
            rotation = rotation_matrix(axis, theta)
        
        # Cylinder parameters
        z = np.linspace(0, length, n)
        theta = np.linspace(0, 2*np.pi, n)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)

        # Apply rotation and translate to center
        xyz = np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])
        xyz_rotated = rotation.dot(xyz)
        x, y, z = xyz_rotated
        x += center[0]
        y += center[1]
        z += center[2]
        points = np.array([[x[i], y[i], z[i]] for i in range(len(x))])
        return points

    def check_collision(self, cylinder):
        grid = self.grid_on_surface()
        for point in grid:
            if cylinder.check_point_inside(point):
                return True
        return False
        
    
    def plot(self, ax):
        # Normalize direction vector
        direction = self.direction
        radius = self.radius
        length = self.height
        center = self.center
        
        # Create rotation matrix to align the z-axis with the direction vector
        z_axis = np.array([0, 0, 1])
        if np.allclose(direction, z_axis):
            rotation = np.eye(3)
        elif np.allclose(direction, -z_axis):
            rotation = rotation_matrix([1, 0, 0], np.pi)
        else:
            axis = np.cross(z_axis, direction)
            theta = np.arccos(np.dot(z_axis, direction))
            rotation = rotation_matrix(axis, theta)
        
        # Cylinder parameters
        z = np.linspace(0, length, 100)
        theta = np.linspace(0, 2*np.pi, 100)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)

        # Apply rotation and translate to center
        xyz = np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])
        xyz_rotated = rotation.dot(xyz)
        x, y, z = xyz_rotated
        x += center[0]
        y += center[1]
        z += center[2]

        # Plot cylinder
        ax.plot_surface(x.reshape(x_grid.shape), y.reshape(y_grid.shape), z.reshape(z_grid.shape), color='k', alpha=0.6)

        # Set labels and plot limits for better visualization
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.axis('equal')

class Cone:
    def __init__(self, center, direction, radius, height):
        self.center = center
        self.direction = direction/np.linalg.norm(direction)
        self.radius = radius
        self.height = height

    def volume(self):
        return np.pi * self.radius**2 * self.height / 3

    def surface_area(self):
        return np.pi * self.radius * (self.radius + np.sqrt(self.radius**2 + self.height**2))
    
    def check_point_inside(self, point):
        center = self.center
        direction = self.direction
        radius = self.radius
        height = self.height
        return np.linalg.norm(point - center) <= radius + height * np.linalg.norm(direction)
    
    def plot(self, ax):
        # Normalize direction vector
        direction = self.direction
        radius = self.radius
        height = self.height
        center = self.center
        
        # Create rotation matrix to align the z-axis with the direction vector
        z_axis = np.array([0, 0, 1])
        if np.allclose(direction, z_axis):
            rotation = np.eye(3)
        else:
            axis = np.cross(z_axis, direction)
            theta = np.arccos(np.dot(z_axis, direction))
            rotation = rotation_matrix(axis, theta)
        
        # Cone parameters
        z = np.linspace(0, height, 100)
        theta = np.linspace(0, 2*np.pi, 100)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * (1 - z_grid / height) * np.cos(theta_grid)
        y_grid = radius * (1 - z_grid / height) * np.sin(theta_grid)

        # Apply rotation and translate to center
        xyz = np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])
        xyz_rotated = rotation.dot(xyz)
        x, y, z = xyz_rotated
        x += center[0]
        y += center[1]
        z += center[2]

        # Plot cone
        ax.plot_surface(x.reshape(x_grid.shape), y.reshape(y_grid.shape), z.reshape(z_grid.shape), color='k', alpha=0.6)

        # Set labels and plot limits for better visualization
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.axis('equal')

class Plane:
    def __init__(self, normal, point):
        self.normal = normal/np.linalg.norm(normal)
        self.point = point
    
    def check_point_inside(self, point):
        return np.dot(self.normal, point - self.point) == 0
    
    def check_point_under(self, point):
        return np.dot(self.normal, point - self.point) <= 0
    
    def check_point_above(self, point):
        return np.dot(self.normal, point - self.point) >= 0
    
    def plot(self, ax):
        # Plane parameters
        normal = self.normal
        point = self.point
        d = np.dot(normal, point)

        # Create a grid for the plane
        if normal[2] != 0:
            xx, yy = np.meshgrid(range(-5, 5), range(-5, 5))
            zz = (-normal[0] * xx - normal[1] * yy + d) / normal[2]
        
        elif normal[1] != 0:
            xx, zz = np.meshgrid(range(-5, 5), range(-5, 5))
            yy = (-normal[0] * xx - normal[2] * zz + d) / normal[1]

        else:
            yy, zz = np.meshgrid(range(-5, 5), range(-5, 5))
            xx = (-normal[1] * yy - normal[2] * zz + d) / normal[0]

        # Plot the plane
        ax.plot_surface(xx, yy, zz, color='k', alpha=0.2)
       

class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def volume(self):
        return 4/3 * np.pi * self.radius**3

    def surface_area(self):
        return 4 * np.pi * self.radius**2
    
    def check_point_inside(self, point):
        return np.linalg.norm(point - self.center) <= self.radius
    
    def plot(self, ax, color='b', alpha=0.2):
        # Sphere coordinates
        radius = self.radius
        center = self.center
        phi, theta = np.mgrid[0.0:2*np.pi:100j, 0.0:np.pi:50j]
        x = radius * np.sin(theta) * np.cos(phi) + center[0]
        y = radius * np.sin(theta) * np.sin(phi) + center[1]
        z = radius * np.cos(theta) + center[2]

        # Plot the surface
        ax.plot_surface(x, y, z, color=color, alpha=alpha)
    
class SemiSphere:
    def __init__(self, center, direction, radius):
        self.center = center
        self.direction = direction/np.linalg.norm(direction)
        self.radius = radius
    
    def volume(self):
        return 2/3 * np.pi * self.radius**3
    
    def surface_area(self):
        return np.pi * self.radius**2
    
    def check_point_inside(self, point):
        center = self.center
        direction = self.direction
        radius = self.radius
        return np.dot(point - center, direction) >= 0 and np.linalg.norm(point - center) <= radius
    
    def plot(self, ax):
        # Normalize direction vector
        direction = self.direction
        radius = self.radius
        center = self.center
        
        # Create rotation matrix to align the z-axis with the direction vector
        z_axis = np.array([0, 0, 1])
        if np.allclose(direction, z_axis):
            rotation = np.eye(3)
        elif np.allclose(direction, -z_axis):
            rotation = rotation_matrix([1, 0, 0], np.pi)
        else:
            axis = np.cross(z_axis, direction)
            theta = np.arccos(np.dot(z_axis, direction))
            rotation = rotation_matrix(axis, theta)
        
        # Cylinder parameters
        z = np.linspace(-np.pi/2, np.pi/2, 100)
        theta = np.linspace(np.pi, 2*np.pi, 100)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.sin(z_grid) * np.cos(theta_grid)
        y_grid = radius * np.sin(z_grid) * np.sin(theta_grid)
        z_grid = radius * np.cos(z_grid)

        # Apply rotation and translate to center
        xyz = np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])
        xyz_rotated = rotation.dot(xyz)
        x, y, z = xyz_rotated
        x += center[0]
        y += center[1]
        z += center[2]

        # Plot cylinder
        ax.plot_surface(x.reshape(x_grid.shape), y.reshape(y_grid.shape), z.reshape(z_grid.shape), color='k', alpha=0.2)

        # Set labels and plot limits for better visualization
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.axis('equal')

class Cube:
    def __init__(self, center, direction, sides):
        self.center = center
        self.direction = direction
        self.height = sides[0]
        self.width = sides[1]
        self.depth = sides[2]
        self.dimensions = np.array([self.height, self.width, self.depth])
    
    def volume(self):
        return self.height * self.width * self.depth
    
    def surface_area(self):
        return 2 * (self.height * self.width + self.height * self.depth + self.width * self.depth)
    
    def check_point_inside(self, point):
        half_height = self.height / 2
        half_width = self.width / 2
        half_depth = self.depth / 2
        center = self.center
        
        # Check if the point is inside the cube
        x, y, z = point
        cx, cy, cz = center
        return (cx - half_width <= x <= cx + half_width and
                cy - half_height <= y <= cy + half_height and
                cz - half_depth <= z <= cz + half_depth)
    
    def point_on_surface(self, density=10):
        half_height = self.height / 2
        half_width = self.width / 2
        half_depth = self.depth / 2

        # Center coordinates
        cx, cy, cz = self.center
        
        # Prepare list to hold points
        points = []

        # Generate points for each face
        # Top and Bottom Faces
        for x in np.linspace(cx - half_width, cx + half_width, density):
            for z in np.linspace(cz - half_depth, cz + half_depth, density):
                points.append((x, cy + half_height, z))  # Top face
                points.append((x, cy - half_height, z))  # Bottom face

        # Front and Back Faces
        for x in np.linspace(cx - half_width, cx + half_width, density):
            for y in np.linspace(cy - half_height, cy + half_height, density):
                points.append((x, y, cz + half_depth))  # Front face
                points.append((x, y, cz - half_depth))  # Back face

        # Left and Right Faces
        for z in np.linspace(cz - half_depth, cz + half_depth, density):
            for y in np.linspace(cy - half_height, cy + half_height, density):
                points.append((cx + half_width, y, z))  # Right face
                points.append((cx - half_width, y, z))  # Left face

        return points

    
    def plot(self, ax):
        direction = self.direction
        center = self.center
        height = self.height
        width = self.width
        depth = self.depth
        
        # Create rotation matrix to align the z-axis with the direction vector
        z_axis = np.array([0, 0, 1])
        if np.allclose(direction, z_axis):
            rotation = np.eye(3)
        elif np.allclose(direction, -z_axis):
            rotation = rotation_matrix([1, 0, 0], np.pi)
        else:
            axis = np.cross(z_axis, direction)
            theta = np.arccos(np.dot(z_axis, direction))
            rotation = rotation_matrix(axis, theta)

        # Half side length
        d = depth / 2.0
        h = height / 2.0
        w = width / 2.0

        # Vertices of the cube
        vertices = np.array([
            [-w, -h, -d],
            [w, -h, -d],
            [w, h, -d],
            [-w, h, -d],
            [-w, -h, d],
            [w, -h, d],
            [w, h, d],
            [-w, h, d]
        ])

        # Rotate and translate vertices
        rotated_vertices = np.dot(vertices, rotation.T) + center

        # Define the list of edges
        faces = [
            [rotated_vertices[j] for j in [0, 1, 2, 3]],
            [rotated_vertices[j] for j in [4, 5, 6, 7]],
            [rotated_vertices[j] for j in [0, 3, 7, 4]],
            [rotated_vertices[j] for j in [1, 2, 6, 5]],
            [rotated_vertices[j] for j in [0, 1, 5, 4]],
            [rotated_vertices[j] for j in [2, 3, 7, 6]]
        ]

        # Create the 3D figure
        cube = Poly3DCollection(faces, facecolors='b', linewidths=1, edgecolors='b', alpha=0.2)
        ax.add_collection3d(cube)


class Base:
    def __init__(self, shape):
        self.shape = shape
    
    def volume(self):
        return self.shape.volume()
    
    def surface_area(self):
        return self.shape.surface_area()
    
    def plot(self, ax):
        self.shape.plot(ax)

    def point_on_surface(self, resolution):
        return self.shape.point_on_surface(resolution)
    

class RevoluteJoint:
    def __init__(self, axis, length, radius, axis2 = 0):
        self.type = "revolute"
        self.axis = axis
        self.axis2 = axis2
        if axis2 == 0:
            self.axis2 = axis 
        self.length = length
        self.radius = radius
        self.theta = 0
    
    def set_params(self, params):
        self.theta = params

    def plot(self, ax, origin, direction):
        direction = direction/np.linalg.norm(direction)
        link_shape = Cylinder(origin, direction, self.radius, self.length)
        link_shape.plot(ax)
        sphere = Sphere(origin + direction * self.length, self.radius)
        sphere.plot(ax)
        
class PrismaticJoint:
    def __init__(self, axis, length, radius, dmax, d = 0):
        self.type = "prismatic"
        self.axis = axis
        self.d = d
        self.dmax = dmax
        self.length = length
        self.radius = radius
    
    def set_params(self, param):
        self.d = param

    def plot(self, ax, origin, direction):
        link_shape = Cylinder(origin, direction, self.radius, self.length)
        link_shape.plot(ax)
        sphere = Sphere(origin + direction * self.length, self.radius)
        sphere.plot(ax, "r")
        link_shape = Cylinder(origin + direction * self.length, direction, self.radius, self.d)
        link_shape.plot(ax)
        sphere = Sphere(origin + direction * (self.d+self.length), self.radius)
        sphere.plot(ax)

class Flip:
    def __init__(self, axis, theta, length, radius, axis2):
        self.axis = axis
        self.axis2 = axis2
        self.length = length
        self.radius = radius
        self.type = "flip"
        self.theta = theta
        
    def plot(self, ax, origin, direction):
        link_shape = Cylinder(origin, direction, self.radius, self.length)
        link_shape.plot(ax)
        sphere = Sphere(origin + direction * self.length, self.radius)
        sphere.plot(ax)

class Robot:
    def __init__(self, base, origin, basis, links, thetas):
        self.base = base
        self.origin = np.array(origin)
        self.basis = {"x" : np.array(basis[0]), "y" : np.array(basis[1]), "z" : np.array(basis[2])} 
        self.n = len([link for link in links if link.type == "revolute" or link.type == "prismatic"])
        self.links = links
        self.thetas = thetas
    
    def add_link(self, link):
        self.links.append(link)
        self.n += 1
    
    def robot_shape(self):
        robot = []
        origin = np.copy(self.origin)
        basis = {"x" : np.copy(self.basis["x"]), "y" : np.copy(self.basis["y"]), "z" : np.copy(self.basis["z"])}
        for link in self.links:
            if link.type == "revolute":
                prev_origin = np.copy(origin)
                origin = origin + np.dot(rotation_matrix(basis[link.axis], link.theta), link.length * basis[link.axis2])
                direction = origin - prev_origin
                shape = Cylinder(prev_origin, direction, link.radius, link.length)
                robot.append(shape)
                for b in basis.keys():
                    basis[b] = np.dot(rotation_matrix(basis[link.axis], link.theta), basis[b])
            elif link.type == "prismatic":
                prev_origin = np.copy(origin)
                direction = basis[link.axis]/np.linalg.norm(basis[link.axis])
                origin = origin + direction  * (link.length + link.d)
                shape = Cylinder(prev_origin, direction, link.radius, link.length + link.d)
                robot.append(shape)
            else:
                prev_origin = origin
                direction = np.dot(rotation_matrix(basis[link.axis], link.theta), basis[link.axis2])
                origin = origin + direction * link.length
                shape = Cylinder(prev_origin, direction, link.radius, link.length)
                robot.append(shape)
                for b in basis.keys():
                    basis[b] = np.dot(rotation_matrix(basis[link.axis], link.theta), basis[b])
        return robot
        
    def check_collision(self, robot, walls):
        for point in walls:
            for shape in robot:
                if shape.check_point_inside(point):
                    return True
        return False
    
    def check_collision2(self, constraint):
        end_effector = self.forward_kinematics()
        out = [constraint(point) for point in end_effector]
        return sum(out)>0

    def self_collision(self, robot):
        for i, shape1 in enumerate(robot[:-2]):
            for shape2 in robot[i+2:]:
                if shape1.check_collision(shape2):
                    return True
        return False
    
    def forward_kinematics(self, only_end_effector = False):
        origin = np.copy(self.origin)
        basis = {"x" : np.copy(self.basis["x"]), "y" : np.copy(self.basis["y"]), "z" : np.copy(self.basis["z"])}
        origins = [np.copy(origin)]
        for link in self.links:
            if link.type == "revolute":
                prev_origin = np.copy(origin)
                origin = origin + np.dot(rotation_matrix(basis[link.axis], link.theta), link.length * basis[link.axis2])
                direction = origin - prev_origin
                for b in basis.keys():
                    basis[b] = np.dot(rotation_matrix(basis[link.axis], link.theta), basis[b])
            elif link.type == "prismatic":
                prev_origin = np.copy(origin)
                direction = basis[link.axis]/np.linalg.norm(basis[link.axis])
                origin = origin + direction  * (link.length + link.d)
            else:
                prev_origin = origin
                direction = np.dot(rotation_matrix(basis[link.axis], link.theta), basis[link.axis2])
                origin = origin + direction * link.length
                for b in basis.keys():
                    basis[b] = np.dot(rotation_matrix(basis[link.axis], link.theta), basis[b])
            origins.append(np.copy(origin))
        if only_end_effector:
            return origin
        return origins

    def jacobian(self):
        delta = 0.00001
        x_eff = self.forward_kinematics(True)
        J = np.zeros((3, self.n))
        i = 0
        for link in self.links:
            if link.type == "revolute":
                link.theta += delta
                x = self.forward_kinematics(True)
                J[:, i] = (x - x_eff)/delta
                link.theta -= delta
                i+=1
            elif link.type == "prismatic":
                link.d += delta
                x = self.forward_kinematics(True)
                J[:, i] = (x - x_eff)/delta
                link.d -= delta
                i+=1
        return J
    
    def singularity_check(self, J):
        singular_values = np.linalg.svd(J, compute_uv=False)
        min_singular_value = np.min(singular_values)
        return abs(min_singular_value)
    
    def manipulability_index(self, J):
        return np.sqrt(np.abs(np.linalg.det(J @ J.T)))

    def plot(self, ax):
        self.base.plot(ax)
        origin = np.copy(self.origin)
        basis = {"x" : np.copy(self.basis["x"]), "y" : np.copy(self.basis["y"]), "z" : np.copy(self.basis["z"])}
        colors = ['r', 'g', 'b']

        for j, b in enumerate(basis):
                ax.quiver(origin[0], origin[1], origin[2], basis[b][0], basis[b][1], basis[b][2], color=colors[j], length=0.5)
        for link in self.links:
            if link.type == "revolute":
                prev_origin = np.copy(origin)
                origin = origin + np.dot(rotation_matrix(basis[link.axis], link.theta), link.length * basis[link.axis2])
                direction = origin - prev_origin
                link.plot(ax, prev_origin, direction)
                for b in basis.keys():
                    basis[b] = np.dot(rotation_matrix(basis[link.axis], link.theta), basis[b])
            elif link.type == "prismatic":
                prev_origin = np.copy(origin)
                direction = basis[link.axis]/np.linalg.norm(basis[link.axis])
                origin = origin + direction  * (link.length + link.d)
                link.plot(ax, prev_origin, direction)
            else:
                prev_origin = origin
                direction = np.dot(rotation_matrix(basis[link.axis], link.theta), basis[link.axis2])
                origin = origin + direction * link.length
                link.plot(ax, prev_origin, direction)
                for b in basis.keys():
                    basis[b] = np.dot(rotation_matrix(basis[link.axis], link.theta), basis[b])
            for j, b in enumerate(basis):
                ax.quiver(origin[0], origin[1], origin[2], basis[b][0], basis[b][1], basis[b][2], color=colors[j], length=0.5)
    
    def interactive_plot(self):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(122, projection='3d')

        self.plot(ax)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(0, 8)
        ax.xaxis._axinfo['grid'].update(color=(1, 1, 1, 0))  # Hide the gridlines by setting alpha to zero
        ax.yaxis._axinfo['grid'].update(color=(1, 1, 1, 0))  # Hide the gridlines by setting alpha to zero

        # Optionally, you can adjust other gridlines
        ax.zaxis.grid(True)  # Show gridlines on the Z axis

        ax.set_box_aspect([1,1,1])

        sliders_ax = []
        sliders = []
        j = 0
        for i in range(len(self.links)):
            if self.links[i].type == "revolute":
                sliders_ax.append(plt.axes([0.1, 0.5 - 0.08 * self.n/2 + 0.08*i, 0.35, 0.04]))
                sliders.append(Slider(sliders_ax[j], f'Theta{j}', 0, 360, valinit=0))
                sliders_ax[j].set_title(f'Theta{j+1}')
                sliders_ax[j].set_xlim(0, 360)
                sliders_ax[j].set_ylim(0, 360)
                sliders_ax[j].set_xticks([0, 90, 180, 270, 360])
                j+=1
            elif self.links[i].type == "prismatic":
                dmax = self.links[i].dmax
                sliders_ax.append(plt.axes([0.1, 0.5 - 0.08 * self.n/2 + 0.08*i, 0.35, 0.04]))
                sliders.append(Slider(sliders_ax[j], f'd{j}', 0, dmax, valinit=0))
                sliders_ax[j].set_title(f'd{j+1}')
                sliders_ax[j].set_xlim(0, dmax)
                sliders_ax[j].set_ylim(0, dmax)
                j+=1

        def update(val):
            params = [slider.val for slider in sliders]
            self.thetas = params
            for link in self.links:
                if link.type == "revolute":
                    link.theta = params.pop(0)*np.pi/180
                elif link.type == "prismatic":
                    link.d = params.pop(0)
            ax.clear()
            self.plot(ax)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_zlim(0, 8)
            ax.xaxis._axinfo['grid'].update(color=(1, 1, 1, 0))  # Hide the gridlines by setting alpha to zero
            ax.yaxis._axinfo['grid'].update(color=(1, 1, 1, 0))  # Hide the gridlines by setting alpha to zero

            # Optionally, you can adjust other gridlines
            ax.zaxis.grid(True)  # Show gridlines on the Z axis

            # ax.set_axis_off()
            ax.set_box_aspect([1,1,1])
            plt.draw()

        for slider in sliders:
            slider.on_changed(update)


        plt.show()
    
    def points_plot(self, reachable_points, constraint, shapes, limits, animate = False):
        params = []
        optimized_joint_angles = [0 for i in range(self.n)]
        error = 0
        manipulability_ind = np.inf
        singularities_index = np.inf
        collisions = 0
        self_collision = 0
        for point in reachable_points:
            def objective_function(params, desired_point):
                i = 0
                for link in self.links:
                    if link.type != "flip":
                        link.set_params(params[i])
                        i+=1
                return np.linalg.norm(self.forward_kinematics(True) - desired_point)

            bounds = []
            desired_point = point
            for link in self.links:
                if link.type == "revolute":
                    bounds.append((-2*np.pi, 2*np.pi))
                elif link.type == "prismatic":
                    bounds.append((0, link.dmax))
            result = minimize(objective_function, optimized_joint_angles, args=(desired_point,), bounds = bounds)
            optimized_joint_angles = result.x
            error += objective_function(optimized_joint_angles, desired_point)
            params.append(optimized_joint_angles)
            robot_shape = self.robot_shape()
            collisions += self.check_collision2(constraint)
            self_collision += self.self_collision(robot_shape)

            J = self.jacobian()
            manipulability_ind = min(self.manipulability_index(J), manipulability_ind)
            singularities_index = min(self.singularity_check(J), singularities_index)
        
        print(error, self.n, manipulability_ind, singularities_index, collisions, self_collision)


        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')

        self.plot(ax)
        ax.plot([point[0] for point in reachable_points], [point[1] for point in reachable_points], [point[2] for point in reachable_points], c='r')
        for shape in shapes:
            shape.plot(ax)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_zlim(limits[4], limits[5])
        ax.xaxis._axinfo['grid'].update(color=(1, 1, 1, 0))  # Hide the gridlines by setting alpha to zero
        ax.yaxis._axinfo['grid'].update(color=(1, 1, 1, 0))  # Hide the gridlines by setting alpha to zero

        # Optionally, you can adjust other gridlines
        ax.zaxis.grid(True)  # Show gridlines on the Z axis

        ax.set_box_aspect([1,1,1])

        sliders_ax = []
        sliders = []
        j = 0
        sliders_ax.append(plt.axes([0.1, 0.02, 0.8, 0.04]))
        sliders.append(Slider(sliders_ax[j], f'Point', 0, len(reachable_points), valinit=0))
        # sliders_ax[j].set_title(f'Points')
        sliders_ax[j].set_xlim(0, len(reachable_points))
        sliders_ax[j].set_ylim(0, len(reachable_points))

        def update(val):
            optimized_joint_angles = params[int(sliders[0].val)]
            i = 0
            for link in self.links:
                if link.type != "flip":
                    link.set_params(optimized_joint_angles[i])
                    i+=1
            desired_point = reachable_points[int(sliders[0].val)]
            ax.clear()
            self.plot(ax)
            ax.plot([point[0] for point in reachable_points], [point[1] for point in reachable_points], [point[2] for point in reachable_points], c='r')
            ax.scatter(desired_point[0], desired_point[1], desired_point[2], c='g')
            for shape in shapes:
                shape.plot(ax)

            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])
            ax.set_zlim(limits[4], limits[5])
            ax.xaxis._axinfo['grid'].update(color=(1, 1, 1, 0))  # Hide the gridlines by setting alpha to zero
            ax.yaxis._axinfo['grid'].update(color=(1, 1, 1, 0))  # Hide the gridlines by setting alpha to zero

            # Optionally, you can adjust other gridlines
            ax.zaxis.grid(True)  # Show gridlines on the Z axis

            # ax.set_axis_off()
            ax.set_box_aspect([1,1,1])
            plt.draw()

        for slider in sliders:
            slider.on_changed(update)

        if animate:
            def increment_slider(val):
                current_val = sliders[0].val
                new_val = (current_val + 1) % len(reachable_points)
                sliders[0].set_val(new_val)

            ani = FuncAnimation(fig, increment_slider, interval=100)
        plt.show()
