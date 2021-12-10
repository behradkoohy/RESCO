#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=bk2g18@soton.ac.uk
#SBATCH --partition=ecsstaff
#SBATCH --account=ecsstaff
#SBATCH --time=18:00:00

module load conda
module load sumo

conda activate py3venv
source venv/bin/activate

export SUMO_HOME=/local/software/sumo/1.9.2/usr

python main.py --agent $1 --tr $2 --log_dir /scratch/bk2g18/logs/
