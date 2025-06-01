import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random, math
from mpl_toolkits.mplot3d import Axes3D
from skimage.draw import line

landmarks = np.load('npy/landmarks.npy')
trajectory = np.load('npy/trajectory.npy')

res = 0.1
margin = 1.0
mins = landmarks.min(axis=0) - margin
maxs = landmarks.max(axis=0) + margin
dims = np.ceil((maxs - mins) / res).astype(int)

grid2d = np.zeros((dims[2], dims[0]), dtype=bool)
inds = ((landmarks - mins) / res).astype(int)
for x, y, z in inds:
    if 0 <= x < dims[0] and 0 <= z < dims[2]:
        grid2d[z, x] = True

fig, ax = plt.subplots()
ax.imshow(~grid2d, cmap='gray')
ax.set_title("Click START then GOAL")
pts = plt.ginput(2, timeout=0)
plt.close(fig)

start2d = (int(pts[0][0]), int(pts[0][1]))
goal2d = (int(pts[1][0]), int(pts[1][1]))

class Node:
    def __init__(self, pt, parent=None):
        self.x, self.z = pt
        self.y = 0
        self.parent = parent

def steer(frm, to, step=10):
    dx, dz = to[0]-frm.x, to[1]-frm.z
    d = math.hypot(dx, dz)
    if d < step:
        return to
    return (frm.x + dx/d*step, frm.z + dz/d*step)

def collision_free2d(p1, p2, grid):
    rr, cc = line(int(p1[1]), int(p1[0]), int(p2[1]), int(p2[0]))
    return not np.any(grid[rr, cc])

nodes = [Node(start2d)]
edges = []
path = []
max_iters = 10000
step = 15
goal_thresh = 10

for _ in range(max_iters):
    rnd = (random.uniform(0, dims[0]-1), random.uniform(0, dims[2]-1))
    nearest = min(nodes, key=lambda n: math.hypot(n.x - rnd[0], n.z - rnd[1]))
    new_pt = steer(nearest, rnd, step)
    new_int = (int(new_pt[0]), int(new_pt[1]))
    if collision_free2d((nearest.x, nearest.z), new_int, grid2d):
        new_node = Node(new_int, nearest)
        nodes.append(new_node)
        edges.append((nearest, new_node))
        if math.hypot(new_node.x - goal2d[0], new_node.z - goal2d[1]) < goal_thresh:
            cur = new_node
            while cur:
                path.append((cur.x, cur.y, cur.z))
                cur = cur.parent
            path.reverse()
            break

fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.set_xlim(0, dims[0])
ax3d.set_ylim(0, 1)
ax3d.set_zlim(0, dims[2])
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.set_title("3D RRT Path Planning")

edge_lines = [ax3d.plot([], [], [], color='gray')[0] for _ in edges]
path_line = ax3d.plot([], [], [], color='blue', linewidth=2)[0]
robot_dot = ax3d.plot([], [], [], 'ro', markersize=6)[0]

def init():
    for ln in edge_lines:
        ln.set_data([], [])
        ln.set_3d_properties([])
    path_line.set_data([], [])
    path_line.set_3d_properties([])
    robot_dot.set_data([], [])
    robot_dot.set_3d_properties([])
    return edge_lines + [path_line, robot_dot]

def animate(i):
    if i < len(edges):
        p, c = edges[i]
        xs, ys, zs = [p.x, c.x], [p.y, c.y], [p.z, c.z]
        edge_lines[i].set_data(xs, ys)
        edge_lines[i].set_3d_properties(zs)
        return [edge_lines[i]]
    else:
        idx = min(i - len(edges), len(path)-1)
        xs = [pt[0] for pt in path[:idx+1]]
        ys = [pt[1] for pt in path[:idx+1]]
        zs = [pt[2] for pt in path[:idx+1]]
        path_line.set_data(xs, ys)
        path_line.set_3d_properties(zs)
        robot_dot.set_data([xs[-1]], [ys[-1]])
        robot_dot.set_3d_properties([zs[-1]])
        return [path_line, robot_dot]

total_frames = len(edges) + len(path)
ani = FuncAnimation(fig3d, animate, init_func=init,
                    frames=total_frames, interval=50, blit=False)

plt.show()