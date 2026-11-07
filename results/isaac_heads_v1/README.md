# isaac_heads_v1: INVALID run (kept for the record)

Produced on TAMU HPRC FASTER (Tesla T4, Isaac Sim 5.1.0 container) on 2026-11-06, by job 3242469.
It is the last of four duplicate jobs that wrote to the same folders, so `data/`, `runs/` and `results/`
are from one consistent run: data written 19:30, model 19:31, results 19:36.

**Do not use these numbers.** The environment had a bug: `IsaacViewpointEnv._load_asset` deleted
`/World/Head` and loaded the next head at the same path, and the RTX renderer kept the first head's
mesh. Every identity was rendered as Lee Perry-Smith's mesh with another head's texture. The three
heads' side outlines overlap at IoU ~0.98. As a result:

- All policies succeed about 50% of the time (summary.csv), and nearly every episode accepts at the first view.
- The held-out identity (lee_perry_smith) is misidentified as "digital_emily_2" in ~90% of episodes.
- Calibration chose thresholds of 0 (runs/isaac_heads_v1/accept.json), because validation held only 2 near-identical identities.

Fixed afterwards in this branch, but not yet run on a GPU:
- isaac_env: one prim per head under /World/Heads/<id>, switched by visibility.
- isaac_preflight: blocks the run if two identities' side silhouettes overlap at IoU > 0.97.
- calibrate: thresholds never go below the config's eval.accept values.

The FaceVerse and Emily preflight grid images are left out on purpose: their licenses forbid public
redistribution (see tools/assets/ATTRIBUTION.md).
