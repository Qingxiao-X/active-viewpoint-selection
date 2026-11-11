### Views

| image | camera az / el / r (read back) | look-at err | key light pos (m) | key I | roughness | spec px face box | spec px head | clipped px face box | face median luma | render settle diff |
|---|---|---|---|---|---|---|---|---|---|---|
| `clean_front.png` | +0.00° / +0.00° / 0.6000 m | 0.000° | (0.632, 0.753, 1.688) | 2.251e+05 | 0.20 | 0.18% | 0.11% | 0.00% | 0.481 | 0.100 |
| `glare_front.png` | +0.00° / +0.00° / 0.6000 m | 0.000° | (1.123, 0.098, 1.41) | 6.752e+05 | 0.20 | 2.68% | 1.78% | 0.01% | 0.749 | 0.073 |
| `glare_left.png` | -15.00° / +0.00° / 0.6000 m | 0.000° | (1.123, 0.098, 1.41) | 6.752e+05 | 0.20 | 2.67% | 1.71% | 0.00% | 0.740 | 0.071 |
| `glare_right.png` | +15.00° / +0.00° / 0.6000 m | 0.000° | (1.123, 0.098, 1.41) | 6.752e+05 | 0.20 | 2.01% | 1.31% | 0.02% | 0.748 | 0.573 |
| `diag_glare_front_rough0.70.png` | +0.00° / +0.00° / 0.6000 m | 0.000° | (1.123, 0.098, 1.41) | 6.752e+05 | 0.70 | 1.06% | 0.73% | 0.00% | 0.759 | 0.390 |

### Camera poses (world frame)

| image | position (m) | quaternion w,x,y,z (Isaac 'world' camera axes: +X fwd, +Z up) |
|---|---|---|
| `clean_front` | (0.6, 0.0, 1.0) | (0.0, 0.0, 0.0, 1.0) |
| `glare_front` | (0.6, 0.0, 1.0) | (0.0, 0.0, 0.0, 1.0) |
| `glare_left` | (0.5796, -0.1553, 1.0) | (0.1305, 0.0, 0.0, 0.9914) |
| `glare_right` | (0.5796, 0.1553, 1.0) | (-0.1305, 0.0, 0.0, 0.9914) |

### ArcFace (reference `arcface_reference_front.png`, detected: True)

| image | face detected | det score | cosine to reference |
|---|---|---|---|
| `clean_front` | True | 0.833 | 0.823 |
| `glare_front` | True | 0.820 | 0.835 |
| `glare_left` | True | 0.808 | 0.837 |
| `glare_right` | True | 0.800 | 0.835 |
| `diag_glare_front_rough0.70` | True | 0.846 | 0.805 |

### Action demo

| step | action | accepted | az (deg) | r (m) | look-at err | min valid depth (m) | center 20px median depth (m) | RGB/depth fg IoU | spec px face box |
|---|---|---|---|---|---|---|---|---|---|
| 0 | START | yes | +0.00 | 0.6000 | 0.000° | 0.477 | 0.496 | 0.994 | 2.87% |
| 1 | LEFT | yes | -15.00 | 0.6000 | 0.000° | 0.481 | 0.509 | 0.995 | 2.42% |
| 2 | LEFT | yes | -30.00 | 0.6000 | 0.000° | 0.492 | 0.507 | 0.994 | 1.06% |
| 3 | LEFT | yes | -45.00 | 0.6000 | 0.000° | 0.502 | 0.512 | 0.994 | 1.18% |
| 4 | LEFT | **rejected** (limit) | -45.00 | 0.6000 | – | – | – | – | – |
| 5 | RIGHT | yes | -30.00 | 0.6000 | 0.000° | 0.492 | 0.507 | 0.994 | 1.02% |
| 6 | RIGHT | yes | -15.00 | 0.6000 | 0.000° | 0.481 | 0.509 | 0.995 | 2.39% |
| 7 | RIGHT | yes | +0.00 | 0.6000 | 0.000° | 0.477 | 0.496 | 0.994 | 3.15% |
| 8 | RIGHT | yes | +15.00 | 0.6000 | 0.000° | 0.482 | 0.508 | 0.994 | 2.08% |
| 9 | FORWARD | yes | +15.00 | 0.5000 | 0.000° | 0.381 | 0.407 | 0.994 | 2.06% |
| 10 | FORWARD | yes | +15.00 | 0.4000 | 0.000° | 0.282 | 0.307 | 0.997 | 1.82% |
| 11 | FORWARD | **rejected** (limit) | +15.00 | 0.4000 | – | – | – | – | – |
| 12 | BACKWARD | yes | +15.00 | 0.5000 | 0.000° | 0.381 | 0.407 | 0.994 | 1.93% |
| 13 | BACKWARD | yes | +15.00 | 0.6000 | 0.000° | 0.482 | 0.508 | 0.994 | 1.93% |
| 14 | BACKWARD | yes | +15.00 | 0.7000 | 0.000° | 0.582 | 0.608 | 0.995 | 1.96% |

### Depth array (glare_front)

```
{
 "shape": [
  480,
  640
 ],
 "dtype": "float32",
 "n_valid": 84331,
 "n_inf": 222869,
 "n_nan": 0,
 "n_nonpositive": 0,
 "min_valid_m": 0.4770650863647461,
 "max_valid_m": 0.6677327156066895,
 "center_20px_median_m": 0.49596166610717773,
 "rgb_depth_foreground_iou": 0.9939518981372318,
 "background_rgb": [
  158.0,
  158.0,
  158.0
 ]
}
```

exposure settings: {"/rtx/rendermode": "RaytracedLighting", "/rtx/post/histogram/enabled": false, "/rtx/post/tonemap/op": 6, "/rtx/post/tonemap/filmIso": 100.0, "/rtx/post/tonemap/cameraShutter": 50.0, "/rtx/post/tonemap/fNumber": 5.0, "/rtx/post/tonemap/whitepoint": [1.0, 1.0, 1.0], "/rtx/post/aa/op": 3, "/rtx/directLighting/sampledLighting/enabled": true}
intrinsics K: [[855.165771484375, 0.0, 320.0], [0.0, 855.165771484375, 240.0], [0.0, 0.0, 1.0]]
key base intensity: 225052.0 | startup s: 24.8 | first frame s: 7.4 | total s: 152.0 | errors: []

### Repeatability vs outputs_run2_pinned_settle30 

| image | mean abs RGB diff (0-255) | spec px face box run1 → this run | ArcFace cos run1 → this run |
|---|---|---|---|
| `clean_front` | 1.216 | 0.11% → 0.18% | 0.786 → 0.823 |
| `glare_front` | 1.091 | 2.27% → 2.68% | 0.850 → 0.835 |
| `glare_left` | 1.162 | 2.33% → 2.67% | 0.840 → 0.837 |
| `glare_right` | 0.899 | 1.99% → 2.01% | 0.823 → 0.835 |
