# Learned viewpoint selection (Active Viewpoint Selection under Specular Glare)

This is the **Learned** policy for the project. It is a supervised action-quality predictor

    q_hat = MLP(current observation features, candidate action features)
    action = argmax over valid a of  q_hat(a) − movement_cost_weight · movement_cost(a)

The package also includes the offline data collector, grouped-split training, and a shared evaluator that runs Random, Greedy, and Learned on paired episodes. Core code (`avs/`) never imports Isaac Sim. Data processing, training, inference and evaluation all run on any machine, using a deterministic **mock environment**.

> **Mock results are software validation only.** The mock imitates the structure of the problem (pose falloff, a light-dependent specular highlight that corrupts the embedding toward a confuser identity). It is not evidence of real face-perception performance.

When this was written the repository was empty: there were no environment, perception, Random/Greedy or evaluator implementations to reuse. The interfaces below are proposals for teammates to plug their code into (see *Plugging in teammates' code*).

## Layout

```
avs/actions.py        Viewpoint, ActionSpace (Δaz/Δel grid, box bounds, masking, arc-length cost)
avs/types.py          PolicyObservation (policy input) vs OracleInfo (privileged); Frame; RenderError
avs/perception.py     InsightFacePerception (pretrained, lazy import), glare/depth metrics,
                      Gallery (observable match vs ground-truth label), build_observation, compute_label
avs/features.py       FeatureSchema (units, missing values), Normalizer (train-only stats + missing flags)
avs/envs/base.py      ViewpointEnv interface (load_scene, move_to, render, get/set_state, oracle, enrollment_frame)
avs/envs/mock_env.py  deterministic mock env + MockPerception (exact state restore)
avs/envs/isaac_env.py Isaac Sim adapter — UNTESTED TEMPLATE
avs/scenes.py         scene randomization ranges, condition bins for grouped reporting
avs/collect.py        offline collector (CLI)
avs/dataset.py        dataset schema v1, IO, grouped splits, leakage checks
avs/model.py, train.py  MLP, checkpoint IO, training with validation-based selection (CLI)
avs/calibrate.py      acceptance thresholds from the VALIDATION split (CLI)
avs/policies.py       Policy interface; random / greedy_heuristic / greedy_preview / learned
avs/evaluate.py       shared evaluator (CLI); avs/plots.py figures
configs/mock.yaml     example config (all knobs);  configs/isaac_example.yaml  Isaac template
tests/                unit + integration tests;    scripts/smoke_test.sh
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

## Commands (mock backend)

Any config value can be overridden with dotted `key=value` arguments, e.g. `collect.n_scenes=50`.

```bash
# smoke test: pytest + small collect → train → calibrate → evaluate (~1 min on CPU)
scripts/smoke_test.sh

# full mock run
.venv/bin/python -m avs.collect   --config configs/mock.yaml              # -> data/mock_v1/
.venv/bin/python -m avs.train     --config configs/mock.yaml              # -> runs/mock_v1/
.venv/bin/python -m avs.calibrate --config configs/mock.yaml --out runs/mock_v1/accept.json
.venv/bin/python -m avs.evaluate  --config configs/mock.yaml eval.accept_file=runs/mock_v1/accept.json
# fresh (never-collected) scenes instead of the held-out test split:
.venv/bin/python -m avs.evaluate  --config configs/mock.yaml eval.scene_source=fresh eval.out_dir=results/mock_fresh
```

Collection can be sharded: run it several times with distinct `collect.start_scene_id` and `collect.out_dir`, then list every directory in `train.datasets`.

## Action space and observation

- **Action space.** The camera sits on a sphere of fixed `radius` (m) and always faces the head center. An action is a (Δaz, Δel) move in degrees from a grid; there is no no-op. Moves that would leave the `az_limits`/`el_limits` box are **masked**, never clipped. Movement cost is the great-circle arc length in meters. Every policy uses the same `ActionSpace`.
- **Policy input.** `PolicyObservation` contains only:
  - camera azimuth/elevation (deg) and radius (m)
  - face detected, detection score, and face box center/size (as fractions of the image)
  - estimated head pose (deg)
  - top-1 similarity and top-1/top-2 margin against the shared enrollment gallery
  - glare fraction and glare centroid within the face box ([-1, 1])
  - valid-depth fraction and depth std (m) within the face box

  The step count is also present but is not a feature. A missing value is `None` → `NaN`. The normalizer fills missing values with the train mean and appends a `<name>__missing` indicator column.
- **Never in policy inputs:** ground-truth identity, lighting/material parameters, scene id, labels, or anything rendered from unvisited viewpoints. `FeatureSchema` rejects these names. The tests check that features are unchanged when oracle metadata changes, and that dataset features read only the `obs.*`/`act.*` columns.
- **Depth.** Valid-depth fraction is a sensing-quality measurement, **not** a liveness signal.

## Training target

The target for a candidate action is measured at the resulting view: `q_id = sim(emb, ref[gt]) − max_{j≠gt} sim(emb, ref[j])`.

- **One enrolled identity.** With `label.mode: auto` and a single identity, the label falls back to `sim(emb, ref[gt])`. In that case identity discrimination (incorrect acceptance) cannot be evaluated. Negatives are never invented.
- **No face detected.** `q_id = label.detection_failure_value` (default −1).
- **Auxiliary terms.** Optional glare/depth terms (`aux_glare_weight`, `aux_depth_weight`) are off by default. Their raw values are always stored separately as `label.aux_*`.
- **Loss.** Huber (default) or MSE.

## Dataset format (schema v1)

A dataset directory contains:

| File | Contents |
|---|---|
| `meta.json` | schema version, config, action space, status counts, `failures`, `restore_checks`, collection cost (renders, seconds) |
| `scenes.jsonl` | one randomized scene per line (**privileged**) |
| `gallery.npz` | shared enrollment embeddings |
| `transitions.npz` | one row per (state, candidate action) |

The columns of `transitions.npz` are grouped by prefix:

| Prefix / column | Contents | Used as features? |
|---|---|---|
| `group.*` | scene_id, state_id, identity | no |
| `obs.*` | current-observation fields | yes |
| `act.*` | candidate-action fields, plus `action_index` | yes |
| `status` | `ok` / `invalid_action` / `render_failure` | — |
| `label.*` | q_id, label, aux_glare, aux_depth | targets only |
| `next.*` | observable confidence and correctness of the resulting view | calibration only |

Only `ok` rows are used for training. Invalid actions and render failures stay as explicit rows with `NaN` labels and are counted. They are never turned into low-quality examples.

## Splits and checkpoints

- **Splits are by scene.** Every start viewpoint and candidate transition of a scene lands in the same split.
- **Held-out identities.** `split.n_heldout_identities` (or an explicit list) adds a `test_heldout_identity` split; those identities never appear in train or val.
- **Train-only fitting.** Normalization statistics come from train only.
- **Checkpoint selection** uses `val_loss` (or `val_regret`). Test metrics are written once for reporting and never used for selection.
- **Checkpoint files:** `model.pt`, `model_config.json`, `feature_schema.json`, `normalizer.json`, `splits.json`, `config.json`, `metrics.json`. `metrics.json` includes per-state decision regret against a random-choice reference.
- **Tune on validation only.** Set `movement_cost_weight`, `revisit_penalty` and the acceptance thresholds from validation (`avs.calibrate` refuses test splits), never from the test set.

## Evaluation protocol (`avs.evaluate`)

- **Shared setup.** Every policy gets the same scenes, start viewpoints, action space, perception, enrollment gallery, budget and acceptance rule.
- **Steps.** Step 0 is the initial view. Each movement consumes one action.
- **Stopping rule.** It uses observable confidence only (detected, `det_score`, top-1 similarity, margin). Ground truth only scores an accepted prediction as correct or incorrect.
- **Outcomes.** Each episode ends as `correct_accept`, `incorrect_accept`, `budget_exhausted`, or `infra_error`. Render failures are counted and excluded from rates.
- **`greedy_preview`** renders every valid neighbor before each move. These are counted as `previews` / `acquisitions`, and the policy is labelled as a **preview baseline** in all outputs. `greedy_heuristic` is a no-preview greedy rule that uses observable glare position and frontality.
- **Learned policy memory.** It keeps only its own episode history: an optional `revisit_penalty` discourages moving back to an already-visited view. The same option applies to the greedy baselines.

Outputs go to `eval.out_dir`:

| File | Contents |
|---|---|
| `episodes.csv`, `episodes.jsonl` | per-episode results (the `.jsonl` also has per-step traces) |
| `summary.json` / `summary.csv` | success, failure, incorrect acceptance, budget exhaustion, steps to success, path length, previews, inference ms vs sensing ms |
| `success_vs_budget.csv` | success rate vs action budget |
| `by_condition.csv` | results by light intensity, light off-axis angle, and roughness |
| `paired.csv` | learned vs each baseline on paired episodes, with a bootstrap 95% CI |
| PNG plots | success vs budget, success by condition, path length |

## Plugging in teammates' code

- **Environment:** subclass/wrap `avs.envs.base.ViewpointEnv` and register it in `avs/envs/__init__.py:make_env`.
- **Perception:** any callable `Frame -> PerceptionResult`.
- **Random/Greedy:** wrap them in `avs.policies.Policy` (`reset(seed)`, `_choose(obs, valid_actions)`) and add them to `build_policy`. If a policy renders candidates, set `uses_preview = True` and call the evaluator-provided `self.previewer(action)` so its acquisitions are counted.

## Integration checklist for the real Isaac Sim experiment (not done yet)

1. Install Isaac Sim 4.5+/5.x plus `insightface` and `onnxruntime-gpu` in the Isaac Python environment, and `pip install -r requirements.txt` there.
2. Fill in `configs/isaac_example.yaml`:
   - USD paths for each head identity
   - the skin shader prim path
   - the head center
   - optionally an arm controller `"module:function"`
3. Check `avs/envs/isaac_env.py`. It has never been run. Each `TODO(integration)` in that file needs to be verified, including:
   - asset orientation (the face must look along +x)
   - the sign convention of the DistantLight rotation
   - the camera look-at
   - the shader input name (`reflection_roughness_constant` vs `roughness`)
4. Test state restoration. Restoration here is deterministic reconstruction, not a renderer snapshot (`exact_restore = False`). Run a short collection with `collect.verify_restore_every` set low and check that `meta.json → restore_checks` shows `max_emb_diff ≈ 0`.
5. Check enrollment. The gallery comes from a frontal capture under neutral lighting (`enroll_light`), and it must detect a face for every identity.
6. Check the glare thresholds (`perception.glare_*`) on a few real renders before collecting.
7. Run collect → train → calibrate → evaluate with `configs/isaac_example.yaml`.
