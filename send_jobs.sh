#for AGENT in "IDQN" "MPLight" "MPLightFULL"; do
for AGENT in "IDQN"; do
	for N in {0,61,66,67,68,69,72,73,74,75,76,77,84,85,86,87,90,91,92,96}; do
		sbatch dist_task.slurm $AGENT $N
	done
done
