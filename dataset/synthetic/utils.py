import numpy as np
import heapq, math, random

def random_pos_variation(position, dx, dy):
    x, y = position
    random_dx = random.uniform(-dx, dx)
    random_dy = random.uniform(-dy, dy)
    new_x = x + random_dx
    new_y = y + random_dy
    return (new_x, new_y)


class NoFreeCellsError(RuntimeError):
    pass

def spawn_goals(
    n, grid, pads,
    x_bounds, y_bounds,
    *, rng=None,
    max_attempts=1000,
    loosen_factor=1.5
):
    if n <= 0:
        return []
    if rng is None:
        rng = np.random.default_rng()

    # free-cell scan
    i_min = max(0, int((x_bounds[0] - grid.x_min) / grid.resolution))
    i_max = min(grid.cols, int(math.ceil((x_bounds[1] - grid.x_min) / grid.resolution)))
    j_min = max(0, int((y_bounds[0] - grid.y_min) / grid.resolution))
    j_max = min(grid.rows, int(math.ceil((y_bounds[1] - grid.y_min) / grid.resolution)))

    free = 0
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            if grid.passable((i, j), pads):
                free += 1
    if free == 0:
        raise NoFreeCellsError(f"No free cells in region x={x_bounds}, y={y_bounds}")

    goals = []
    attempts = 0
    attempts_since_loosen = 0
    total_cap = max_attempts * 10
    cur_xb, cur_yb = x_bounds, y_bounds

    while len(goals) < n:
        if attempts_since_loosen >= max_attempts:
            # expand bounds
            cx = (cur_xb[0] + cur_xb[1]) / 2
            cy = (cur_yb[0] + cur_yb[1]) / 2
            half_w = (cur_xb[1] - cur_xb[0]) * loosen_factor / 2
            half_h = (cur_yb[1] - cur_yb[0]) * loosen_factor / 2
            cur_xb = (cx - half_w, cx + half_w)
            cur_yb = (cy - half_h, cy + half_h)
            print(f"Loosening bounds to x={cur_xb}, y={cur_yb}")
            attempts_since_loosen = 0

        gx = rng.uniform(*cur_xb)
        gy = rng.uniform(*cur_yb)
        idx = grid.to_idx((gx, gy))
        if grid.in_bounds(idx) and grid.passable(idx, pads):
            goals.append((gx, gy))
            attempts_since_loosen = 0
        else:
            attempts_since_loosen += 1

        attempts += 1
        if attempts > total_cap:
            raise RuntimeError(
                f"Unable to place {n} goals after {attempts} attempts "
                f"(initial x={x_bounds}, y={y_bounds})."
            )
    return goals


class Grid:
    def __init__(self, x_bounds, y_bounds, resolution):
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.resolution = resolution
        self.cols = int((self.x_max-self.x_min)/resolution)
        self.rows = int((self.y_max-self.y_min)/resolution)
        
    def to_idx(self, pt):
        return (int((pt[0]-self.x_min)/self.resolution),
                int((pt[1]-self.y_min)/self.resolution))
    def to_coord(self, idx):
        return (self.x_min + idx[0]*self.resolution + self.resolution/2,
                self.y_min + idx[1]*self.resolution + self.resolution/2)
    def in_bounds(self, idx):
        return 0 <= idx[0] < self.cols and 0 <= idx[1] < self.rows
    def passable(self, idx, pads):
        pt = self.to_coord(idx)
        return all(not point_in_patch(pt, p) for p in pads)

# ── Theta* ──────────────────────────────────────────────────────
def theta_star(start, goal, grid, pads, los_step=0.05):
    s_idx, g_idx = grid.to_idx(start), grid.to_idx(goal)
    if not grid.passable(s_idx, pads) or not grid.passable(g_idx, pads): return []

    dirs = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    g = {s_idx: 0}; parent = {s_idx: s_idx}
    los_edge = {} 
    h0 = math.hypot(*(np.subtract(grid.to_coord(s_idx), grid.to_coord(g_idx))))
    open_set = [(h0, s_idx)]

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == g_idx:
            # reconstruct
            idx_path = []
            i = cur
            while i != parent[i]:
                idx_path.append(i)
                i = parent[i]
            idx_path.append(s_idx)
            idx_path.reverse()

            # densify only LOS edges
            dense = [grid.to_coord(idx_path[0])]
            for u, v in zip(idx_path, idx_path[1:]):
                a = grid.to_coord(u); b = grid.to_coord(v)
                if los_edge.get(v, False):
                    # densify
                    dist = math.hypot(b[0]-a[0], b[1]-a[1])
                    n = max(1, int(dist/los_step))
                    for t in [i/n for i in range(1, n+1)]:
                        dense.append((a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1])))
                else:
                    dense.append(b)
            return dense

        for dx, dy in dirs:
            nb = (cur[0]+dx, cur[1]+dy)
            if not grid.in_bounds(nb) or not grid.passable(nb, pads): 
                continue
            parent_coord = grid.to_coord(parent[cur])
            nb_coord     = grid.to_coord(nb)
            cur_coord    = grid.to_coord(cur)

            if line_of_sight(parent_coord, nb_coord, pads, grid.resolution/2):
                new_g = g[parent[cur]] + math.hypot(*(np.subtract(parent_coord, nb_coord)))
                used_los = True
                candidate_parent = parent[cur]
            else:
                new_g = g[cur] + math.hypot(*(np.subtract(cur_coord, nb_coord)))
                used_los = False
                candidate_parent = cur

            if nb not in g or new_g < g[nb]:
                g[nb] = new_g
                parent[nb] = candidate_parent
                los_edge[nb] = used_los
                f = new_g + math.hypot(*(np.subtract(nb_coord, grid.to_coord(g_idx))))
                heapq.heappush(open_set, (f, nb))
    return []


def smooth_path_with_beziers(path, radius=0.2, resolution=20):
    if len(path) < 3:
        return path

    smoothed_path = []
    last_point = np.array(path[0])
    smoothed_path.append(tuple(last_point))

    for i in range(1, len(path) - 1):
        p_prev = np.array(path[i-1])
        p_turn = np.array(path[i])
        p_next = np.array(path[i+1])

        v_prev = p_prev - p_turn
        v_next = p_next - p_turn

        dist_prev = np.linalg.norm(v_prev)
        dist_next = np.linalg.norm(v_next)
        
        effective_radius = min(radius, dist_prev / 2, dist_next / 2)

        v_prev_norm = v_prev / dist_prev
        v_next_norm = v_next / dist_next

        arc_start = p_turn + v_prev_norm * effective_radius
        arc_end = p_turn + v_next_norm * effective_radius

        smoothed_path.append(tuple(arc_start))

        t_values = np.linspace(0, 1, resolution)
        for t in t_values:
            point = (1-t)**2 * arc_start + 2*t*(1-t) * p_turn + t**2 * arc_end
            smoothed_path.append(tuple(point))
        
        last_point = arc_end

    smoothed_path.append(path[-1])
    return smoothed_path


def point_in_transformed_rectangle_fast(pt, rect_data):
    """Fast point-in-rectangle test using vector math instead of ray casting"""
    x, y = pt
    corners = rect_data['corners']
    
    # Quick bounding box check first
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    if x < min_x or x > max_x or y < min_y or y > max_y:
        return False
    
    # For rectangles, we can use a more efficient method
    # Check if point is on the same side of each edge
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    # Check against all 4 edges of the rectangle
    d1 = sign(pt, corners[0], corners[1])
    d2 = sign(pt, corners[1], corners[2])
    d3 = sign(pt, corners[2], corners[3])
    d4 = sign(pt, corners[3], corners[0])
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)
    
    return not (has_neg and has_pos)

def point_in_patch(pt, patch_data):
    """Optimized collision detection"""
    if patch_data['type'] == 'rectangle':
        return point_in_transformed_rectangle_fast(pt, patch_data)
    elif patch_data['type'] == 'circle':
        cx, cy = patch_data['center']
        radius = patch_data['radius']
        return math.hypot(pt[0]-cx, pt[1]-cy) <= radius
    return False

def line_of_sight(p1, p2, pads, step):
    dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    n = int(dist/step)
    for i in range(1, n+1):
        t = i/n
        x = p1[0] + t*(p2[0]-p1[0]); y = p1[1] + t*(p2[1]-p1[1])
        if any(point_in_patch((x, y), p) for p in pads):
            return False
    return True


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtrans
import math, random

# ── Enhanced geometry helpers ────────────────────────────────────

def create_padded(
    ax,
    position,
    width,
    height,
    color,
    edgecolor,
    *,
    pad=0.1,
    rounding=0.15,
    zorder=1,
    rotation_deg=0,
    jitter_x=(0, 0),
    jitter_y=(0, 0),
    draw=True
):
    """
    Draw OR just compute a padded rectangle with optional jitter and rotation.

    Returns both 'collision_data' (with padding) and
    'original_collision_data' (without padding).
    """

    # 1) jitter
    x0 = position[0] + random.uniform(*jitter_x)
    y0 = position[1] + random.uniform(*jitter_y)

    # center of the ORIGINAL rect
    cx, cy = x0 + width/2, y0 + height/2

    # --- compute original (no-pad) collision data ---
    orig_corners = [
        (x0,          y0),
        (x0 + width,  y0),
        (x0 + width,  y0 + height),
        (x0,          y0 + height)
    ]
    if rotation_deg != 0:
        ang = math.radians(rotation_deg)
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        def rot(pt):
            x, y = pt
            xr = (x - cx) * cos_a - (y - cy) * sin_a + cx
            yr = (x - cx) * sin_a + (y - cy) * cos_a + cy
            return (xr, yr)
        orig_corners = [rot(pt) for pt in orig_corners]

    original_collision_data = {
        'type': 'rectangle',
        'corners': orig_corners,
        'original_center': (cx, cy),
        'width': width,
        'height': height,
        'cumulative_translation': (0, 0),
        'cumulative_rotation': rotation_deg
    }

    # --- compute padded collision data ---
    pad_w, pad_h = width + 2*pad, height + 2*pad
    pad_x, pad_y = cx - pad_w/2, cy - pad_h/2

    pad_corners = [
        (pad_x,             pad_y),
        (pad_x + pad_w,     pad_y),
        (pad_x + pad_w,     pad_y + pad_h),
        (pad_x,             pad_y + pad_h)
    ]
    if rotation_deg != 0:
        pad_corners = [rot(pt) for pt in pad_corners]

    collision_data = {
        'type': 'rectangle',
        'corners': pad_corners,
        'original_center': (cx, cy),
        'width': pad_w,
        'height': pad_h,
        'cumulative_translation': (0, 0),
        'cumulative_rotation': rotation_deg
    }

    # If not drawing, just return both data dicts
    if ax is None or not draw:
        return {
            "patch": None,
            "pad": None,
            "outline": None,
            "center": (cx, cy),
            "original_collision_data": original_collision_data,
            "collision_data": collision_data
        }

    # --- drawing branch ---
    patch = patches.Rectangle((x0, y0), width, height,
                              linewidth=1,
                              edgecolor=edgecolor,
                              facecolor=color,
                              alpha=0.8,
                              zorder=zorder)
    transform = mtrans.Affine2D().rotate_deg_around(cx, cy, rotation_deg) + ax.transData
    patch.set_transform(transform)
    ax.add_patch(patch)

    pad_rect = patches.Rectangle((pad_x, pad_y), pad_w, pad_h,
                                 linewidth=0,
                                 facecolor="none",
                                 transform=transform,
                                 zorder=zorder - 0.1)
    outline = patches.FancyBboxPatch((pad_x, pad_y), pad_w, pad_h,
                                     boxstyle=f"round,pad=0.,rounding_size={rounding}",
                                     linewidth=1,
                                     edgecolor='black',
                                     facecolor='none',
                                     transform=transform,
                                     zorder=zorder - 0.1)
    ax.add_patch(outline)

    return {
        "patch": patch,
        "pad": pad_rect,
        "outline": outline,
        "center": (cx, cy),
        "original_collision_data": original_collision_data,
        "collision_data": collision_data
    }


def get_bounds(center=None, width=None, height=None, margin=0.3, 
               margin_x=None, margin_y=None, shape="rectangle"):
    
    """
    Generate bounds either around an object or from explicit bounds.
    
    Args:
        center: (x, y) tuple of the object center
        width: width of the object
        height: height of the object
        margin: uniform distance from object edges to create goal bounds
        margin_x: specific margin for x-direction (overrides margin)
        margin_y: specific margin for y-direction (overrides margin)
        shape: "rectangle" or "circle" - shape of the bounds area
    
    Returns:
        dict with x_bounds and y_bounds tuples
    """

    # Otherwise, calculate bounds from center and dimensions
    if center is None or width is None or height is None:
        raise ValueError("Must provide either (center, width, height) or (x_bounds, y_bounds)")
    
    obj_x, obj_y = center
    
    # Use specific margins if provided, otherwise use uniform margin
    mx = margin_x if margin_x is not None else margin
    my = margin_y if margin_y is not None else margin
    
    if shape == "rectangle":
        # Create rectangular bounds that surround the object
        x_bounds = (
            obj_x - width/2 - mx,
            obj_x + width/2 + mx
        )
        y_bounds = (
            obj_y - height/2 - my,
            obj_y + height/2 + my
        )
    
    elif shape == "circle":
        # Create circular bounds - use the larger dimension + margin as radius
        radius = max(width, height)/2 + max(mx, my)
        x_bounds = (obj_x - radius, obj_x + radius)
        y_bounds = (obj_y - radius, obj_y + radius)
    
    return {"x_bounds": x_bounds, "y_bounds": y_bounds}
