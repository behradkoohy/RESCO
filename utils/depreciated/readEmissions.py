import os
import xml.etree.ElementTree as ET
import json

log_dir = '/Users/behradkoohy/sumo-scratchpad/RESCO/logs/'
ems_dir = '/Users/behradkoohy/sumo-scratchpad/RESCO/util/emissions/'
env_base = '..'+os.sep+'environments'+os.sep
names = [folder for folder in next(os.walk(log_dir))[1]]

metrics = [ 'CO_abs',
            'CO2_abs',
            'HC_abs',
            'PMx_abs',
            'NOx_abs',
            'fuel_abs'
        ]

# co_tot, co2_tot, hc_tot, pmx_tot, nox_abs, fuel_abs = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
co, co2, hc, pmx, nox, fuel = [], [], [], [], [], [] 

total_emissions_by_name = {name : {} for name in names}

for name in names:
    trip_emissions = {metric : [] for metric in metrics}

    for i in range(1,10000):
        if i % 10 == 0: print(i)
        trip_file_name = log_dir+name + os.sep + 'tripinfo_'+str(i)+'.xml'
        emis_values = {metric : 0.0 for metric in metrics}
        if not os.path.exists(trip_file_name):
            print('No '+trip_file_name)
            break
        try:
            tree = ET.parse(trip_file_name)
            root = tree.getroot()
            n_cars = 0
            for child in root:
                try:
                    emission = child[0].attrib
                    for metric in metrics:
                        emis_values[metric] = emis_values[metric] + float(emission[metric])
                        n_cars += 1

                except Exception as e:
                    raise e
                    break
            for metric in metrics:
                trip_emissions[metric] = trip_emissions.get(metric, []) + [emis_values[metric]/float(n_cars)]
        except Exception as e:
            raise e
            break
    
    total_emissions_by_name[name] = trip_emissions

with open("total_emissions.py", "r+") as f:
    data = f.read()
    f.seek(0)
    f.write("emission = {")
    for name in names:
        f.write("\"" + name + "\" : " + json.dumps(total_emissions_by_name[name]) + ",\n")
    f.write("}")