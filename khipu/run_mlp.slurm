#!/bin/bash
#SBATCH --job-name=cpd-mlp
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=experiment_khipu.log
#SBATCH --error=experiment_khipu.err

# Job para el experimento del proyecto CPD #7 en Khipu (nodo CPU de 32 cores).
# Limita BLAS a 1 hilo/proceso: la unica paralelizacion es la nuestra.

set -e
cd $HOME/cpd_mlp/code
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate mlp

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RESULTS_FILE=$HOME/cpd_mlp/data/results_khipu.csv
export RESULTS_DIR=$HOME/cpd_mlp/data
export N_VALUES=5000,10000,20000,40000
export P_VALUES=1,2,4,8,16,32
export REPS=2

echo "=== Job CPD-MLP en Khipu ==="
echo "nodo: $(hostname)   cpus visibles: $(nproc)   $(date)"
echo "python: $(which python)   sklearn: $(python -c 'import sklearn; print(sklearn.__version__)')"

python run_experiments.py

echo "=== FIN $(date) ==="
