import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random, math
from mpl_toolkits.mplot3d import Axes3D
from skimage.draw import line
# Change pgm file name accordingly 
map_img = cv2.imread('pgm/map3d-2.pgm', cv2.IMREAD_GRAYSCALE)
grid2d = (map_img == 0)
H, W = grid2d.shape

fig2d, ax2d = plt.subplots()
ax2d.imshow(~grid2d, cmap='gray')
ax2d.set_title("Click START then GOAL")
pts = plt.ginput(2, timeout=0)
plt.close(fig2d)
start2d = (int(pts[0][0]), int(pts[0][1]))
goal2d  = (int(pts[1][0]), int(pts[1][1]))

Y = 1
grid3d = np.zeros((H, Y, W), dtype=bool)
for y in range(Y):
    grid3d[:, y, :] = grid2d

start = (start2d[0], 0, start2d[1])
goal  = (goal2d[0], 0, goal2d[1])

class Node3D:
    def __init__(self, pt, parent=None):
        self.x, self.y, self.z = pt
        self.parent = parent

def steer(frm, to, step):
    dx, dz = to[0] - frm.x, to[2] - frm.z
    d = math.hypot(dx, dz)
    if d < step:
        return to
    return (frm.x + dx/d*step, 0, frm.z + dz/d*step)

def collision(p1, p2, grid):
    rr, cc = line(int(p1[2]), int(p1[0]), int(p2[2]), int(p2[0]))
    for z, x in zip(rr, cc):
        if grid[z, 0, x]:
            return False
    return True

nodes = [Node3D(start)]
edges = []
path = []
found = False
max_iters = 10000
step = 10
goal_thresh = 5

for i in range(max_iters):
    rnd = (random.uniform(0, W-1), 0, random.uniform(0, H-1))
    nearest = min(nodes, key=lambda n: math.hypot(n.x-rnd[0], n.z-rnd[2]))
    new_pt = steer(nearest, rnd, step)
    new_int = (int(new_pt[0]), 0, int(new_pt[2]))
    if collision((nearest.x, nearest.y, nearest.z), new_int, grid3d):
        new_node = Node3D(new_pt, nearest)
        nodes.append(new_node)
        edges.append((nearest, new_node))
        if math.hypot(new_node.x-goal[0], new_node.z-goal[2]) < goal_thresh:
            found = True
            cur = new_node
            while cur:
                path.append((cur.x, cur.y, cur.z))
                cur = cur.parent
            path.reverse()
            break

fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.set_xlim(0, W)
ax3d.set_ylim(0, 1)
ax3d.set_zlim(0, H)

edge_lines = [ax3d.plot([], [], [], color='gray')[0] for _ in edges]
path_line, = ax3d.plot([], [], [], color='blue', linewidth=2)
robot_dot, = ax3d.plot([], [], [], 'ro', markersize=6)

def init():
    for ln in edge_lines:
        ln.set_data([], [])
        ln.set_3d_properties([])
    path_line.set_data([], [])
    path_line.set_3d_properties([])
    robot_dot.set_data([], [])
    robot_dot.set_3d_properties([])
    return edge_lines + [path_line, robot_dot]

def animate(frame):
    if frame < len(edges):
        p, c = edges[frame]
        xs, ys, zs = [p.x, c.x], [p.y, c.y], [p.z, c.z]
        edge_lines[frame].set_data(xs, ys)
        edge_lines[frame].set_3d_properties(zs)
        return [edge_lines[frame]]
    else:
        idx = min(frame - len(edges), len(path) - 1)
        xs = [pt[0] for pt in path[:idx+1]]
        ys = [pt[1] for pt in path[:idx+1]]
        zs = [pt[2] for pt in path[:idx+1]]
        path_line.set_data(xs, ys)
        path_line.set_3d_properties(zs)
        robot_dot.set_data(xs[-1:], ys[-1:])
        robot_dot.set_3d_properties(zs[-1:])
        return [path_line, robot_dot]

total = len(edges) + len(path)
ani = FuncAnimation(fig3d, animate, init_func=init, frames=total, interval=50, blit=False)

plt.show()