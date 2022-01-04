import matplotlib.pyplot as plt
import numpy as np

from total_emissions import emission
from graph import map_title, alg_name

metrics = [ 'CO_abs',
            'CO2_abs',
            'HC_abs',
            'PMx_abs',
            'NOx_abs',
            'fuel_abs'
        ]

fs = 21

maps_done = set()

for metric in metrics:
    fig, ax = plt.subplots()

    labels = ('0', '20', '40', '60', '80', '100')

    ax.set(title=metric + " : " + map_title['ingolstadt21'])
    ax.set(xlabel="Iteration", ylabel=metric)

    points = [0, 20, 40, 60, 80, 100]
    labels = ('0', '20', '40', '60', '80', '100', '..1400')
    # ax.set_yticks()
    ax.set_xticks(points)

    for key, value in emission.items():
        print(key)
        split = key.split("-")
        algo = split[0]
        if algo not in ('STOCHASTIC', 'IDQN', 'MPLight'):
            continue
        envir = split[2]
        maps_done.add(envir)

        ax.plot((value[metric]), label=algo)
    ax.legend(loc=0)
    plt.show()




