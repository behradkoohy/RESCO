#for AGENT in "IDQN" "MPLight" "MPLightFULL"; do
for AGENT in "IDQN"; do
	for N in {0..20}; do
		sbatch dist_task.slurm $AGENT $N
	done
done
