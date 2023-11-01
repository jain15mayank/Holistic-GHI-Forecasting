import matplotlib.pyplot as plt
import numpy as np
import json

latent_dim = 32
# Results Path
resultsFolder = './results/Conv2/'
jsonFilePath = resultsFolder+'loss_log_'+str(latent_dim)+'.json'

training_logs = {'epoch'        : np.array([]),
                 'loss'         : np.array([])}#,
                #  'val_loss'     : np.array([])}

# deleteRowsAfterEpoch = 489

# json_lines = []
# with open(jsonFilePath, mode='rt', buffering=1) as open_file:
#     for line in open_file.readlines():
#         j = json.loads(line)
#         if int(j['epoch']) < deleteRowsAfterEpoch:
#             json_lines.append(line)

# with open(jsonFilePath, mode='wt', buffering=1) as open_file:
#     open_file.writelines(json_lines)


with open(jsonFilePath, mode='rt', buffering=1) as logsJSON:
    lines = logsJSON.readlines()
    for line in lines:
        logsDict = json.loads(line)
        training_logs['epoch'] = np.append(training_logs['epoch'], logsDict['epoch'])
        training_logs['loss'] = np.append(training_logs['loss'], logsDict['loss'])
        # training_logs['val_loss'] = np.append(training_logs['val_loss'], logsDict['val_loss'])

# Plot Loss Curves
plt.figure(figsize=(10, 6))
plt.rc('font', size=17)         # controls default text sizes
plt.rc('axes', titlesize=19)    # fontsize of the axes title
plt.rc('axes', labelsize=19)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=17)   # fontsize of the tick labels
plt.rc('ytick', labelsize=17)   # fontsize of the tick labels
plt.rc('legend', fontsize=17)   # legend fontsize
plt.rc('figure', titlesize=19)  # fontsize of the figure title
plt.plot(training_logs['epoch'], training_logs['loss'], 'r--', label='Training Loss')
# plt.plot(training_logs['epoch'], training_logs['val_loss'], 'b--', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(resultsFolder+'training_loss_characteristics_'+str(latent_dim)+'.pdf', bbox_inches = 'tight', pad_inches = 0.05)
plt.close()