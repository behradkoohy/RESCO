import sqlite3
import matplotlib.pyplot as plt
from db_config import db_name, metrics, emission_outputs

all_inputs = metrics + emission_outputs

conn = sqlite3.connect(db_name)
cur = conn.cursor()

cur.execute("SELECT DISTINCT algorithm, state, reward, environment FROM experiments;")
resu = cur.fetchall()
# import pdb
# pdb.set_trace()
print(resu)
algorithms = [x[0] for x in resu]
states = [x[1] for x in resu]
rewards = [x[2] for x in resu]

# print(resu, algorithms, states)

algo_labels = {
    "IDQN":"IDQN",
    "SRainbow":"Rainbow",
    "Rainbow":"Rainbow",
    "MPLight": ""
}

results_averaged = {}

for metric in all_inputs:
    for (algorithm, state, reward, env) in resu:
        cur.execute("SELECT max(epoch) FROM experiments WHERE metric=? AND algorithm=? AND state=? AND environment=? AND reward=?;", (metric, algorithm, state, env, reward))
        final_epoch = cur.fetchone()[0]
        for x in range(1, final_epoch+1):
            cur.execute("SELECT avg(result) FROM experiments WHERE epoch=? AND metric=? AND algorithm=? AND state=? AND environment=? AND reward=?;", (x, metric, algorithm, state, env, reward))
            avg = cur.fetchall()[0]
            # print(avg)
            if avg[0] is None:
                print((x, (algorithm, state, reward, metric)))
                continue
            results_averaged[(algorithm, state, reward, metric)] = results_averaged.get((algorithm, state, reward, metric), []) + [avg]


# for key, val in results_averaged.items():
#     print(len(val) ,key, val)
# exit()

for metric in all_inputs:
    for (algorithm, state, reward, env) in resu:
        if algorithm in ['MPLight', 'SRainbow']:
            continue
    # for algorithm in algorithms:
        # print(algorithm[0])
        plt.plot(results_averaged[(algorithm, state, reward, metric)], label=algo_labels[algorithm])
    # plt.legend([r[0] for r in resu], bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    plt.legend([algo_labels[r[0]] for r in resu], bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    # plt.title(metric)
    plt.ylabel('Duration')
    plt.xlabel('Epoch')
    # plt.show()
    plt.savefig(db_name.replace(".db", "") + "_" + metric + "epochs" + '.png', bbox_inches='tight')

