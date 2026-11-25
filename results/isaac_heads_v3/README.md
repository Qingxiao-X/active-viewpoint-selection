# isaac_heads_v3: dual-light experiment (negative result, kept for the record)

Produced 2026-11-18/19 on TAMU HPRC Grace (Tesla T4, same container as v2), jobs 20664811
(collect/train) and 20673481 (recalibrate/evaluate at sim>=0.65/margin>=0.25).

**Do not treat this as the main result; see `results/isaac_heads_v2/` instead.** v3 adds a
second, independently-randomized directional light (`avs/scenes.py` `fill_light_*`,
`avs/envs/isaac_env.py` `FillLight` prim) to test whether a harder, less predictable glare
pattern would make `greedy_heuristic` (which only reacts to the *one* detected glare blob's
position) fail more often than `learned`. It did not:

- At the same threshold (sim>=0.65/margin>=0.25) that gave v2 a 94.4%-vs-100% gap, v3 gives
  `random`=98.6% vs `learned`/`greedy_heuristic`/`greedy_preview`=100% -- the dual-light
  scenes are no harder on average (`by_condition.csv`: `fill_light=dual` and `fill_light=single`
  success rates are statistically indistinguishable for every policy).
- `learned` remains perfectly tied with the greedy baselines in every run attempted (v2 and
  v3 alike): with a fairly dense 15-degree action grid and a budget of 6 moves, the policy
  almost always has an escape view clear of whichever light source(s) are active, so a
  second independent point light doesn't create new failure modes the simple
  "move away from the glare centroid" heuristic can't already handle.

A first attempt (before this one) placed the fill light's azimuth in the head's back/side
range (90..270 deg), which never overlapped what the camera can actually reach (+-75 deg) --
that version showed *zero* difference between dual- and single-light scenes, as expected.
This version fixed the azimuth range to overlap the camera's reachable range and still found
no meaningful separation; see `data/isaac_heads_v3/scenes.jsonl` + `transitions.npz` if
re-analyzing.

**Possible follow-ups, not attempted here:** correlate the two lights to be adversarial by
construction (e.g. place the fill light to specifically re-illuminate wherever moving away
from the key light's glare would otherwise look), or reduce `eval.budget` so a suboptimal
first move is costlier instead of changing the lighting model.
