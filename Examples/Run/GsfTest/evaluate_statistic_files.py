import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

path = "."

if len(sys.argv) == 2:
    path = sys.argv[1]

failure_file = os.path.join(path, "gsf-fail-statistics.csv")
success_file = os.path.join(path, "gsf-success-statistics.csv")

assert os.path.exists(failure_file)
assert os.path.exists(success_file)

failure_data = pd.read_csv(failure_file)
success_data = pd.read_csv(success_file)


#eta_failure = -np.log(np.tan(failure_data["theta"].to_numpy()))
#eta_success = -np.log(np.tan(success_data["theta"].to_numpy()))

#plt.hist(eta_failure,50, histtype="step")
#plt.hist(eta_success,50, histtype="step")
#plt.hist(np.concatenate([eta_failure, eta_success]), 50, histtype="step")
#plt.legend(["failure", "success", "all"])
#plt.show()


plt.hist(failure_data["theta"],50, histtype="step")
plt.hist(success_data["theta"],50, histtype="step")
plt.hist(np.concatenate([failure_data["theta"], success_data["theta"]]), 50, histtype="step")
plt.legend(["failure", "success", "all"])
plt.show()



#plt.scatter(success_data["initialMomentum"], success_data["fwdSteps"])
#plt.yscale('log')
#plt.xscale('log')
#plt.xlabel("initialMomentum")
#plt.ylabel("steps")
#plt.show()


#plt.scatter(success_data["initialMomentum"], success_data["fwdPathlength"])
##plt.yscale('log')
##plt.xscale('log')
#plt.xlabel("initialMomentum")
#plt.ylabel("fwdPathlength")
#plt.show()

#plt.hist(failure_data["initialMomentum"], 50, label="failure", histtype="step")
#plt.title("failure")
#plt.show()
#plt.hist(success_data["initialMomentum"], 50, label="success", histtype="step")
#plt.title("success")
#plt.show()
#all_momenta = np.concatenate([failure_data["initialMomentum"].to_numpy(), success_data["initialMomentum"]])
#print("Num samples: {}".format(len(all_momenta)))
#plt.hist(all_momenta, 50, label="all", histtype="step")
#plt.title("all momenta")
#plt.show()
