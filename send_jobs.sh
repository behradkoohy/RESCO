#for AGENT in "IDQN" "MPLight" "MPLightFULL"; do
for AGENT in "IDQN" "IDQN400" "RAINBOW"; do
	for N in {1..10}; do
		sbatch dist_task.slurm $AGENT $N
	done
done
