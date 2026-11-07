#!/bin/bash
#SBATCH -J rebuild_isaac
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/user/qingxiao/689/hprc/logs/%x-%j.out
# Rebuild the Isaac Sim image so it runs as an unprivileged Singularity user, with the
# perception deps (insightface, onnxruntime-gpu, pyyaml, matplotlib) baked in.
set -euo pipefail
module load WebProxy/0000
H=/scratch/user/qingxiao/689/hprc
L=${TMPDIR:-/tmp}/rb_$SLURM_JOB_ID; mkdir -p $L/tmp $L/cache
export SINGULARITY_TMPDIR=$L/tmp SINGULARITY_CACHEDIR=$L/cache
SB=$L/sb
singularity build --sandbox $SB $H/isaac-sim_5.1.0.sif
chmod -R u+rwX,go+rX $SB 2>/dev/null || true
mkdir -p $SB$H $SB/scratch/user/qingxiao/689
mkdir -p $SB/isaac-sim/kit/cache $SB/isaac-sim/kit/data $SB/isaac-sim/kit/logs $SB/opt/avs_cache
PY="singularity exec --writable --no-home -B $H:$H $SB /isaac-sim/python.sh"
echo "== bundled versions"
$PY - <<'PYEOF' | tee $L/versions.txt
import importlib, sys
print("python", sys.version.split()[0])
for mod, dist in [("numpy","numpy"),("scipy","scipy"),("cv2","opencv-python"),("PIL","pillow"),("matplotlib","matplotlib"),
                  ("torch","torch"),("yaml","pyyaml"),("sklearn","scikit-learn"),("skimage","scikit-image"),
                  ("onnx","onnx"),("onnxruntime","onnxruntime"),("requests","requests"),("tqdm","tqdm"),("packaging","packaging")]:
    try:
        m = importlib.import_module(mod); print("PKG", dist, getattr(m, "__version__", "?"), getattr(m, "__file__", ""))
    except Exception as e:
        print("MISSING", dist, type(e).__name__, e)
PYEOF
# pin everything already present so pip never replaces Isaac's own copies
grep '^PKG' $L/versions.txt | awk '$3 != "?" && $2 != "opencv-python" {print $2"=="$3}' > $L/constraints.txt
grep -q '^PKG opencv-python' $L/versions.txt && HAVE_CV2=1 || HAVE_CV2=0
echo "constraints:"; cat $L/constraints.txt; echo "have cv2: $HAVE_CV2"
EXTRA="pyyaml matplotlib scikit-image onnx easydict prettytable tqdm requests"
[ $HAVE_CV2 = 0 ] && EXTRA="$EXTRA opencv-python-headless<4.11"
$PY -m pip install --no-cache-dir -c $L/constraints.txt "onnxruntime-gpu[cuda,cudnn]>=1.21,<1.23" $EXTRA
$PY -m pip install --no-cache-dir --no-deps $H/wheels/insightface-0.7.3-cp311-cp311-linux_x86_64.whl
echo "== import check"
$PY -c "import numpy, insightface, onnxruntime, yaml, matplotlib, skimage, cv2; from insightface.app import FaceAnalysis; print('numpy', numpy.__version__, 'ort', onnxruntime.__version__, onnxruntime.get_available_providers(), 'cv2', cv2.__version__)"
chmod -R u+rwX,go+rX $SB 2>/dev/null || true
cp $L/versions.txt $H/logs/isaac_versions.txt
singularity build --force $H/isaac-sim_5.1.0_avs.sif $SB
ls -la $H/isaac-sim_5.1.0_avs.sif
rm -rf $L
