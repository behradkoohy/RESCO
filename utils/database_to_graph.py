import sqlite3
import matplotlib.pyplot as plt
from db_config import db_name, metrics, emission_outputs

all_inputs = metrics + emission_outputs

conn = sqlite3.connect(db_name)
cur = conn.cursor()

cur.execute("SELECT DISTINCT algorithm FROM experiments;")
algorithms = cur.fetchall()

results_averaged = {}

for metric in all_inputs:
    for algorithm in algorithms:
        cur.execute("SELECT max(epoch) FROM experiments WHERE epoch=100 AND metric=? AND algorithm=?;", (metric, algorithm[0]))
        final_epoch = cur.fetchone()[0]
        for x in range(1, final_epoch+1):
            cur.execute("SELECT avg(result) FROM experiments WHERE epoch=? AND metric=? AND algorithm=?;", (x, metric, algorithm[0]))
            avg = cur.fetchall()[0]
            # print(avg)
            if avg[0] is None:
                print((x, metric, algorithm[0]))
                continue
            results_averaged[(algorithm[0], metric)] = results_averaged.get((algorithm[0], metric), []) + [avg]


# for key, val in results_averaged.items():
#     print(len(val) ,key, val)

for metric in all_inputs:
    for algorithm in algorithms:
        plt.plot(results_averaged[(algorithm[0], metric)], label=algorithm[0])
    plt.legend([algorithm[0] for algorithm in algorithms])
    plt.title(metric)
    plt.show()

