#for AGENT in "IDQN" "MPLight" "MPLightFULL"; do
#for AGENT in "IDQN" "IDQN_average_speed" "IDQN_average_speed_norm" "IDQN_mwait" "IDQN_wait" "IDQN_wait_norm" "IDQN_pressure" "IDQN_queue" "IDQN_queue_sq" "IDQN_pressure_sq" "IDQN_queue_maxwait" "IDQN_queue_maxwait_neighbourhood"; do
#for AGENT in "IDQN" "IDQN_mwait" "IDQN_wait" "IDQN_wait_norm" "IDQN_queue" "IDQN_queue_sq" "IDQN_queue_maxwait" "IDQN_queue_maxwait_neighbourhood"; do
for AGENT in "STOCHASTIC"; do
	for N in {1..20}; do
		sbatch noalpha_dist_task.slurm $AGENT $N
	done
done
