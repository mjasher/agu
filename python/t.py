import numpy as np

d=np.load("/home/mikey/Downloads/openturns_results.npz")

average_run_time=d["average_run_time"]
average_surrogate_run_time=d["average_surrogate_run_time"]
building_time=d["building_time"]
sample_YPC=d["sample_YPC"]
sample_X=d["sample_X"]
sample_Y=d["sample_Y"]
SUT_by_output=d["SUT_by_output"]
SU_by_output=d["SU_by_output"]



import json
import csv
print sample_X.shape
print sample_Y.shape
print sample_YPC.shape
print average_run_time
print average_surrogate_run_time
print average_run_time/average_surrogate_run_time
print SUT_by_output

# import json
# data_dir = "/home/mikey/Dropbox/pce/agu/data/"
# with open(data_dir+'scatter.json','w') as f:
# 	f.write(json.dumps({
# 		"sample_X": sample_X.tolist(),
# 		"sample_Y": sample_Y.tolist(),
# 		"sample_YPC": sample_YPC.tolist(),
# 		"average_run_time": average_run_time,
# 		"average_surrogate_run_time": average_surrogate_run_time,
# 		"SUT_by_output": SUT_by_output.tolist(),
# 		"SU_by_output": SU_by_output.tolist(),
# 		}))