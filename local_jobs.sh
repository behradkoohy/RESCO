#for AGENT in "IDQN" "MPLight" "MPLightFULL"; do
for AGENT in "GRAPHP"; do
	for N in {0..5}; do
		python main.py --agent $AGENT --tr $N
	done
done
