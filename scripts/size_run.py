"""Pick collection / evaluation sizes that fit the remaining batch-job time.

    python size_run.py collect <preflight_report.json> <seconds_left>      -> prints n_scenes
    python size_run.py eval <preflight_report.json> <splits.json> <seconds_left> -> prints max_scenes_per_split

Estimates use the render+perception time measured by the preflight. They are
deliberately conservative: a run cut by the time limit loses the step in progress.
"""

import json
import sys

STARTS = 3                 # collect.starts_per_scene
VALID_PER_START = 12.5     # mean valid candidate actions (15 minus masked moves near the box edge)
EPISODES_PER_SCENE = 3
RENDERS_PER_EPISODE_SET = 50   # random + heuristic + learned (<=7 each) + preview (<=7 + 6*~12)
PROC_STARTUP_S = 240           # Isaac startup + enrollment per python process
TRAIN_CALIB_S = 600
MARGIN_S = 900


def per_render(rep):
    return float(rep["timing"]["render_plus_perception_s"]) * 1.1


def collect_size(rep, left):
    t = per_render(rep)
    switch = float(rep["timing"].get("asset_switch_s", 2.0))
    col_scene = STARTS * (1 + VALID_PER_START) * t + 0.7 * switch
    avail = left - 2 * PROC_STARTUP_S - TRAIN_CALIB_S - MARGIN_S
    best = 30
    for n in range(30, 401, 3):
        n_test = 0.15 * (2 / 3) * n + min(n / 3, 15)  # test split + held-out identity scenes (capped)
        ev = n_test * EPISODES_PER_SCENE * RENDERS_PER_EPISODE_SET * t
        if n * col_scene + ev <= avail:
            best = n
    return best


def eval_size(rep, splits, left):
    t = per_render(rep)
    avail = left - PROC_STARTUP_S - MARGIN_S
    sizes = {k: len(v) for k, v in splits["splits"].items() if k.startswith("test")}
    for cap in range(30, 0, -1):
        n = sum(min(cap, s) for s in sizes.values())
        if n * EPISODES_PER_SCENE * RENDERS_PER_EPISODE_SET * t <= avail:
            return cap
    return 1


if __name__ == "__main__":
    mode = sys.argv[1]
    rep = json.load(open(sys.argv[2]))
    if mode == "collect":
        print(collect_size(rep, float(sys.argv[3])))
    else:
        print(eval_size(rep, json.load(open(sys.argv[3])), float(sys.argv[4])))
