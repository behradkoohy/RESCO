import os
import sqlite3
import xml.etree.ElementTree as ET
import re

import concurrent

from db_config import emission_outputs, metrics, db_name

log_dir = '/home/behradkoohy/sumo_scratchpad/behrad-resco/RESCO/logs/'
# log_dir = '/media/behradkoohy/pdata/iridis/agent_experiments/didqn_exp/'
db_dir = "outputs.dir"
env_base = '..' + os.sep + 'environments' + os.sep

titles = set([folder.split("-")[0] for folder in os.listdir(log_dir)])
conn = sqlite3.connect(db_name)

def create_connection(database):
    """ create a database connection to a SQLite database """
    conn = sqlite3.connect(database)
    c = conn.cursor()
    c.execute('''
	          CREATE TABLE IF NOT EXISTS experiments (
	          metric TEXT NOT NULL,
	          algorithm TEXT NOT NULL,
	          environment TEXT NOT NULL,
	          trial INT NOT NULL,
	          epoch INT NOT NULL,
	          result FLOAT NOT NULL,
	          PRIMARY KEY (metric, algorithm, environment, trial, epoch)
	          )
	          ''')
    conn.commit()
    c.close()

mproc = False

def get_cur(database):
    # conn = sqlite3.connect(database)
    return conn.cursor()

def update_database_values(metric, algorithm, environment, trial, epoch, value):
    conn = get_cur(db_name)
    conn.execute("""
        INSERT OR REPLACE INTO experiments (metric, algorithm, environment, trial, epoch, result)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (metric, algorithm, environment, trial, epoch, value))


# Aggregate all the relevant information into lists of folders
experiments = {}
for experiment_name in titles:
    experiments[experiment_name] = experiments.get(experiment_name, []) + [x for x in os.listdir(log_dir) if experiment_name in x]


def add_to_db(trial, run_name):
    run_name_split = run_name.split("-")
    algorithm = run_name_split[0]
    environment = run_name_split[2]
    # trial is already computed

    xml_regex = re.compile("tripinfo_.*\.xml")
    trip_files = list(filter(xml_regex.match, os.listdir(log_dir + run_name)))
    for trip_file in trip_files:
        epoch = int(trip_file.replace("tripinfo_", "").replace(".xml", ""))
        trip_file_path = log_dir + run_name + os.sep + trip_file
        if not os.path.exists(trip_file_path):
            print('No ' + trip_file_path)
            continue
        try:
            tree = ET.parse(trip_file_path)
            root = tree.getroot()
            number_trips, total = 0, 0.0
            metric_outs = {}
            # last departure time would go here
            for child in root:
                try:
                    number_trips += 1
                    emission = child[0].attrib
                    # take the emission metrics
                    for met in emission_outputs:
                        metric_outs[met] = metric_outs.get(met, 0) + float(emission[met])
                    # take the performance metrics
                    for met in metrics:
                        metric_outs[met] = metric_outs.get(met, 0) + float(child.attrib[met])
                except Exception as e:
                    print(e)
                    break
            # print(number_trips, metric_outs)
            for name, total in metric_outs.items():
                metric_outs[name] = metric_outs[name] / float(number_trips)
                update_database_values(name, algorithm, environment, trial, epoch, metric_outs[name])
            # print(number_trips, metric_outs)
            # get_cur(db_name).commit()
            conn.commit()

        except ET.ParseError as e:
            print(e)
            break

calculated_outputs = {}

executor = concurrent.futures.ProcessPoolExecutor()

# initialise the database
create_connection(db_name)
print(experiments)
for experiment in experiments:
    unsorted_experiments = [(int(x.split("-")[1].replace("tr", "")), x) for x in experiments[experiment]]
    unsorted_experiments.sort(key=lambda tup: tup[0])
    for trial, run_name in unsorted_experiments:
        print(run_name)
        add_to_db(trial, run_name)
        if mproc:
            executor.submit(add_to_db(trial, run_name))
        else:
            add_to_db(trial, run_name)


    # create_connection("test")

# if __name__ == '__main__':
# 	pass
