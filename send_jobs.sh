#for AGENT in "Rainbow" "MPLight" "MPLightFULL"; do
for AGENT in "Rainbow" "Rainbow_average_speed" "Rainbow_average_speed_norm" "Rainbow_mwait" "Rainbow_wait" "Rainbow_wait_norm" "Rainbow_pressure" "Rainbow_queue" "Rainbow_queue_sq" "Rainbow_pressure_sq" "Rainbow_queue_maxwait" "Rainbow_queue_maxwait_neighbourhood"; do
#for AGENT in "Rainbow" "Rainbow_mwait" "Rainbow_wait" "Rainbow_wait_norm" "Rainbow_queue" "Rainbow_queue_sq" "Rainbow_queue_maxwait" "Rainbow_queue_maxwait_neighbourhood"; do
# for AGENT in "STOCHASTIC"; do
	for N in {1..20}; do
		sbatch noalpha_dist_task.slurm $AGENT $N
	done
done