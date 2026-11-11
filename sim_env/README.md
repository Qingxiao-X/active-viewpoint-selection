# sim_env: Isaac Sim face-glare environment

An Isaac Sim 5.1 scene with one scanned head (PBR skin), a dome light, one key light, and an RGB-D camera that orbits the head with 4 actions (LEFT/RIGHT/FORWARD/BACKWARD). `REPORT.md` describes the scene, the measurements, and the known issues. `outputs/` holds the reference run (job 20431930) to compare against. The raw `.npy` depth and embedding files are left out to keep the repo small.

## Layout

| Path | What it is |
|---|---|
| `configs/glare_scene_pinned.json` | Scene settings used for the reference run: material, lights, camera, render. **Start here.** |
| `configs/glare_scene.json` | Same, but calibrates the key-light intensity at run time |
| `scripts/run_capture.py` | Builds the scene, renders all captures, writes `run_record.json`, and exports `scene/face_glare_scene.usda` |
| `glare/rig.py` | Orbit camera, actions, and limits (pure numpy, no Isaac) |
| `glare/figures.py`, `scripts/summarize.py` | Contact sheets, depth visualizations, and tables (no GPU) |
| `scene/face_glare_scene.usda` | Exported stage. It can be opened in Isaac Sim / USD Composer to inspect the scene. |
| `assets/lee_perry_smith/` | Head USD and textures, CC BY 3.0 (see the license file) |
| `assets_tools/` | Scripts that converted the raw scans to USD |
| `hprc/run_glare.sbatch` | Slurm job for a Grace RTX 6000 node |

## Where to make changes

**Lights.** Lights are built near the top of `scripts/run_capture.py` (search for `# lights`):
- `/World/EnvLight`: a `DomeLight`, set by `dome_light` in the config
- `/World/KeyLight`: a `SphereLight`. Its position is az/el/dist around the head center, set by `key_light` in the config.

To add a light, define another `UsdLux` light there (for example `UsdLux.RectLight.Define(stage, "/World/FillLight")`), and add its settings to the config so they are recorded in `run_record.json`.

**Head material.** The skin shader is the asset's `UsdPreviewSurface` at `/World/Head/Looks/Skin/Shader`. `set_material()` in `run_capture.py` sets `roughness`, `metallic`, and `ior` from `material` in the config. Any other `UsdPreviewSurface` input works the same way: `specularColor`, `clearcoat`, `clearcoatRoughness`, `opacity`, and so on. The scan's specular map (`Map-SPEC.jpg`) is not used yet. For an MDL material such as `OmniSurface` with subsurface scattering, bind a new material to `/World/Head` instead.

**Head asset.** Set `head_usd` in the config. Relative paths are resolved from `sim_env/`. Only Lee Perry-Smith is in the repo: the FaceVerse and Digital Emily licenses forbid redistribution, so ask Qingxiao for those files and **do not commit them**.

**Keep these in mind:**
- The camera uses fixed exposure (auto-exposure off), so new lights change image brightness directly.
- RTX real-time needs about 240 frames to settle after a light moves. The reference run used 120 (`render.settle_frames`). See REPORT §7.

## Run

Run from this folder on a Grace login node:

```bash
cd sim_env
CFG=$PWD/configs/glare_scene_pinned.json OUT=$PWD/outputs_mytest sbatch --export=ALL hprc/run_glare.sbatch
```

- `H` points to the folder with the Isaac Sim container (`isaac-sim_5.1.0_avs.sif`, about 7 GB) and `insightface/` models. The default is Qingxiao's scratch folder. If you can't read it, copy the container to your own scratch space and pass `H=/path/to/that/folder`.
- `RT` (default `$SCRATCH/avs_isaac_runtime`) holds your Kit caches.
- The job writes `scene/face_glare_scene.usda` again, with an absolute path to the head. Don't commit that change unless you mean to.
