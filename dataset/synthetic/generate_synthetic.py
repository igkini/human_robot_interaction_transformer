import json
import math
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scripts.dataset.synthetic.utils import (
    Grid,
    spawn_goals,
    theta_star,
    smooth_path_with_beziers,
    create_padded,
    get_bounds,
    NoFreeCellsError
)
from shapely.geometry import Polygon, Point

# ───────────────────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────────────────
NUM_PARTS   = 5
CSV_PATH    = "paths.csv"
SAVE_EVERY  = 100  
DRAW        = True 

# ───────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────
x_min, y_min, x_max, y_max = -5.25, -3.5, 5.35, 9.0

MOBILE_WIDTH, MOBILE_HEIGHT = 1.75, 0.9
AGV_WIDTH, AGV_HEIGHT       = 1.75, 3.0
TOOL_BASE_WIDTH, TOOL_BASE_HEIGHT = 1.0, 0.5
BATTERY_WIDTH, BATTERY_HEIGHT     = 1.2, 0.5
DYDYNAMIC_CENTER= (0, 4)
DYNAMIC_WIDTH, DYNAMIC_HEIGHT = 1.0, 1.

TOOL_VAR_X, TOOL_VAR_Y = 1.5, 0.5
BATTERY_VAR_X = 0.5

TOOL_ROT_RANGE = 90
AGV_ROT_RANGE  = 10
MOBILE_ROT_RANGE = 10

tasks = json.load(open(Path("scripts/dataset/synthetic/task_library.json")))

# ───────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT + PARTITION
# ───────────────────────────────────────────────────────────────────────────────
def build_environment(ax, agv_key, mobile_key, draw=True):
    pads = []
    objects_no_pad = []

    def cp(ax, *args, **kwargs):
        return create_padded(ax, *args, draw=draw, **kwargs)

    # Dynamic object
    dynamic = cp(ax, DYDYNAMIC_CENTER, DYNAMIC_WIDTH, DYNAMIC_HEIGHT,
                 color="lightgrey", edgecolor="black",
                 jitter_x=(-0.5, 0.5), jitter_y=(-0.5, 0.5),
                 rotation_deg=random.uniform(-10, 10))
    pads.append(dynamic["collision_data"])
    objects_no_pad.append(dynamic["original_collision_data"])

    # Desk
    desk = cp(ax, (4.1 - 1.25, -3.5), 2.5, 7.0, color="wheat", edgecolor="brown")
    pads.append(desk["collision_data"])
    objects_no_pad.append(desk["original_collision_data"])

    # Battery
    battery = cp(ax, (-0.350, -0.850), BATTERY_WIDTH, BATTERY_HEIGHT,
                 color="lightblue", jitter_x=(0, BATTERY_VAR_X), edgecolor="blue")
    pads.append(battery["collision_data"])
    objects_no_pad.append(battery["original_collision_data"])

    # Tool
    tool = cp(ax, (-0.250, 1.400), TOOL_BASE_WIDTH, TOOL_BASE_HEIGHT,
              color="lightgreen", edgecolor="green",
              jitter_x=(0, TOOL_VAR_X),
              jitter_y=(-TOOL_VAR_Y, TOOL_VAR_Y * 0.1),
              rotation_deg=random.uniform(-TOOL_ROT_RANGE, TOOL_ROT_RANGE))
    pads.append(tool["collision_data"])
    objects_no_pad.append(tool["original_collision_data"])

    # AGV
    agv_center = {"random": (-4.5, 0), "marriage_point_agv": (-4.5, -0.5)}[agv_key]
    agv_jitter_y = (-3, 5) if agv_key == "random" else (0., 1.)
    agv = cp(ax, agv_center, AGV_WIDTH, AGV_HEIGHT,
             color="lightcoral", edgecolor="red",
             jitter_y=agv_jitter_y, jitter_x=(0., 0.8),
             rotation_deg=random.uniform(-AGV_ROT_RANGE, AGV_ROT_RANGE))
    pads.append(agv["collision_data"])
    objects_no_pad.append(agv["original_collision_data"])

    # Mobile
    if mobile_key == "assembly":
        mob_pos = (battery["center"][0] - 0.6, battery["center"][1] - 1.395)
        mob = cp(ax, mob_pos, MOBILE_WIDTH, MOBILE_HEIGHT,
                 color="plum", edgecolor="purple",
                 jitter_x=(-0.5, 0.3), jitter_y=(-0.2, 0.2),
                 rotation_deg=random.uniform(-MOBILE_ROT_RANGE, MOBILE_ROT_RANGE))
    else:
        mob_pos = (agv["center"][0] + 0.33 * AGV_WIDTH, agv["center"][1])
        mob = cp(ax, mob_pos, MOBILE_WIDTH, MOBILE_HEIGHT,
                 color="plum", edgecolor="purple",
                 rotation_deg=90 + random.uniform(-MOBILE_ROT_RANGE, MOBILE_ROT_RANGE),
                 jitter_y=(-0.5, 0.5), jitter_x=(-0.1, 0.1))
    pads.append(mob["collision_data"])
    objects_no_pad.append(mob["original_collision_data"])

    centers = tool["center"], agv["center"], mob["center"], battery["center"]
    return pads, centers, objects_no_pad


def partition_goals(entries, n_parts):
    partitions = [[] for _ in range(n_parts)]
    for entry in entries:
        n = entry["n_goals"]
        slice_size = int(round(n * (1 / n_parts)))
        p = [slice_size for _ in range(n_parts - 1)]
        p.append(n - sum(p))
        for i in range(n_parts):
            partitions[i].append((entry["goal"], p[i]))
    return partitions

# ───────────────────────────────────────────────────────────────────────────────
# PATH GENERATION + CSV LOGGING
# ───────────────────────────────────────────────────────────────────────────────
def generate_paths_and_log(ax, starts, goals, grid, pads, writer,
                           task_id, mobile_center, agv_center, grid_id,
                           path_id_start=0, save_every=SAVE_EVERY,
                           draw=True):
    pid = path_id_start
    for sx, sy in starts:
        for gx, gy in goals:
            raw = theta_star((sx, sy), (gx, gy), grid, pads)
            if not raw:
                continue
            path = smooth_path_with_beziers(raw)
            if draw and ax is not None:
              px, py = zip(*path)
              ax.plot(px, py, lw=1.2)

            last_idx = len(path) - 1
            for i, (x, y) in enumerate(path):
                if (i % save_every == 0) or (i == last_idx):
                    writer.writerow([pid, task_id, x, y,
                                     mobile_center[0], mobile_center[1],
                                     agv_center[0], agv_center[1], grid_id])
            pid += 1
    return pid

def generate_occupancy_grid(grid, objects_no_pad, grid_id):
    occupancy = np.zeros((grid.rows, grid.cols), dtype=int)
    
    for obj in objects_no_pad:
        poly = Polygon(obj['corners'])
        
        indices = [grid.to_idx((x, y)) for x, y in obj['corners']]
        i_min = max(0, min(i for i, j in indices))
        i_max = min(grid.cols - 1, max(i for i, j in indices))
        j_min = max(0, min(j for i, j in indices))
        j_max = min(grid.rows - 1, max(j for i, j in indices))
        
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                x = grid.x_min + i * grid.resolution + grid.resolution / 2
                y = grid.y_min + j * grid.resolution + grid.resolution / 2
                if poly.contains(Point(x, y)):
                    occupancy[j, i] = 1
    
    with open(f"grid_{grid_id}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for j in range(grid.rows - 1, -1, -1):
            writer.writerow(list(occupancy[j]))
    
    return grid_id

# ───────────────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────────────
def main():
    path_id_counter = 0
    grid_id_counter = 0

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path_id","task_id","x","y",
                         "mobile_cx","mobile_cy","agv_cx","agv_cy","grid_id"])

        for task in tasks:
            start_parts = partition_goals(task["start_pos"], NUM_PARTS)
            goal_parts  = partition_goals(task["goal_pos"], NUM_PARTS)

            for stage in range(NUM_PARTS):
                ax = None
                if DRAW:
                    fig, ax = plt.subplots(figsize=(9,7))
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.3)
                    ax.set_title(f"Task {task['task_id']} - Part {stage+1}/{NUM_PARTS}: {task['title']}")

                while True:
                    try:
                        pads, (tool_c, agv_c, mob_c, batt_c), objects_no_pad = \
                            build_environment(ax, task["agv_pos"], task["mobile_pos"], draw=DRAW)
                        grid = Grid((x_min,x_max),(y_min,y_max),resolution=0.2)
                        grid_id = f"{task['task_id']}_{stage+1}"
                        generate_occupancy_grid(grid, objects_no_pad, grid_id)
                        grid_id_counter += 1

                        GOAL_AREA = {
                            "random":           get_bounds((0.05,2.75),10.6,12.5,0),
                            "tool_station":     get_bounds(tool_c,1.05,0.55,0.0),
                            "battery_assembly": get_bounds((batt_c[0],batt_c[1]+0.45), batt_c[0]+0.6,0.0,0.05),
                            "agv_ph1":          get_bounds((agv_c[0]+2,agv_c[1]+3.5),1.5,1.5,0.5),
                            "agv_ph2":          get_bounds(agv_c, AGV_WIDTH+0.2, AGV_HEIGHT+0.2,0.0),
                        }

                        # spawn goals
                        starts, goals = [], []
                        for goal_type,n in start_parts[stage]:
                            if n>0:
                                area = GOAL_AREA[goal_type]
                                pts = spawn_goals(n, grid, pads, area["x_bounds"], area["y_bounds"])
                                starts.extend(pts)
                        for goal_type,n in goal_parts[stage]:
                            if n>0:
                                area = GOAL_AREA[goal_type]
                                pts = spawn_goals(n, grid, pads, area["x_bounds"], area["y_bounds"])
                                goals.extend(pts)

                        # logging
                        start_count = len(starts)
                        goal_count = len(goals)
                        print(f"[Task {task['task_id']} - Part {stage+1}] Generated {start_count} starts, {goal_count} goals.")
                        break

                    except NoFreeCellsError as e:
                        print(f"{e} → rebuilding environment and retrying …")
                        if DRAW and ax is not None:
                            ax.cla()
                            ax.set_xlim(x_min, x_max)
                            ax.set_ylim(y_min, y_max)
                            ax.set_aspect('equal')
                            ax.grid(True, alpha=0.3)
                            ax.set_title(f"Task {task['task_id']} - Part {stage+1}/{NUM_PARTS}: {task['title']}")
                        continue

                # plot markers
                if DRAW and ax is not None:
                    if starts: ax.plot(*zip(*starts), 'bo', ms=4, label='start')
                    if goals:  ax.plot(*zip(*goals),  'rx', ms=5, label='goal')

                # paths + logging
                path_id_counter = generate_paths_and_log(
                    ax, starts, goals, grid, pads, writer,
                    task_id=task['task_id'],
                    mobile_center=mob_c,
                    agv_center=agv_c,
                    grid_id=grid_id,
                    path_id_start=path_id_counter,
                    save_every=SAVE_EVERY,
                    draw=DRAW
                )

                if DRAW and ax is not None:
                    ax.legend(loc='upper right')
                    plt.tight_layout()
                    plt.show()

    print(f"Done. Paths saved to {CSV_PATH}")

if __name__ == '__main__':
    main()
