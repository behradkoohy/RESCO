#!/bin/bash

#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bk2g18@soton.ac.uk

module load conda
module load sumo

conda activate py3venv
source venv/bin/activate

export SUMO_HOME=/local/software/sumo/1.9.2/usr

#python main.py --agent MPLight --map ingolstadt21 --trials 1 --libsumo True --procs 1
python main.py --agent MPLightFULL --map ingolstadt21 --trials 1 --libsumo True --procs 1
#python main.py --agent IDQN --map ingolstadt21  --trials 1 --procs 4
