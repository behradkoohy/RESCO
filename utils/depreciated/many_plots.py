import os

log_dir = "/home/behradkoohy/sumo_scratchpad/iridis_logs/"

names = [folder for folder in next(os.walk(log_dir))[1]]
for name in names:
	short_name = ""
	if name.split("-")[0] == "IDQN":
		short_name = "IDQN"
	elif name.split("-")[0] == "MPLight" and name.split("-")[4] == "mplight_full":
		short_name = "MPLight_full"
	elif name.split("-")[0] == "MPLight" and name.split("-")[4] == "mplight":
		short_name = "MPLight"
	folder_contents = [folder for folder in next(os.walk(log_dir + name))[2]] 

	# print(len(folder_contents))

	if len(folder_contents) < 200:
		print(name, len(folder_contents))

print(len(names))