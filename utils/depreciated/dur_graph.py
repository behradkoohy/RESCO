from avg_duration import durations
import matplotlib.pyplot as plt

for x in [x for x in durations if "_yerr" in x]:
	plt.plot(durations[x], label=x)

plt.legend()
plt.show()
print("done")