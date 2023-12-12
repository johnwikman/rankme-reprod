

import mlflow
import matplotlib.pyplot as plt
import numpy as np

runs_data = []

runs = mlflow.search_runs()
runs = runs[runs['status'] == 'FINISHED']
simclr_runs = runs[runs['params.trainer'] == 'simclr']
print(simclr_runs.columns)
# Extract desired metrics and parameters from each run
for _, run in runs.iterrows():
    run_data = {
            'run_id': run['run_id'],
            'rankme_rank': run['metrics.rankme_rank'],
            'CIFAR100_accuracy': run['metrics.CIFAR100_accuracy'],
            'caltech101_accuracy': run['metrics.caltech101_accuracy'],
            'CIFAR100_rank': run['metrics.CIFAR100_rank'],
            'caltech101_rank': run['metrics.caltech101_rank'],
            'trainer': run['params.trainer']
        }
    runs_data.append(run_data)

color_map = {
    'vicreg': 'red',
    'simclr': 'blue',
}

vicreg_data = [run for run in runs_data if run['trainer'] == 'vicreg']
simclr_data = [run for run in runs_data if run['trainer'] == 'simclr']

def plot_linear_regression(data, color, label):
    if data:
        x = [run['rankme_rank'] for run in data]
        y = [run['CIFAR100_accuracy'] + run['caltech101_accuracy'] for run in data]  # Combine both accuracies
        coeffs = np.polyfit(x, y, 1)
        regression_line = np.poly1d(coeffs)(np.unique(x))
        plt.plot(np.unique(x), regression_line, color=color, linestyle='dashed', label=label)

# Plot scatter for 'vicreg'
for run in vicreg_data:
    #plt.scatter(run['rankme_rank'], run['CIFAR100_accuracy'], color=color_map['vicreg'])
    plt.scatter(run['rankme_rank'], run['caltech101_accuracy'], color=color_map['vicreg'])

# Plot scatter for 'simclr'
for run in simclr_data:
    #plt.scatter(run['rankme_rank'], run['CIFAR100_accuracy'], color=color_map['simclr'])
    plt.scatter(run['rankme_rank'], run['caltech101_accuracy'], color=color_map['simclr'])

# Plot linear regressions
plot_linear_regression(vicreg_data, color_map['vicreg'], 'vicreg Regression')
plot_linear_regression(simclr_data, color_map['simclr'], 'simclr Regression')
plot_linear_regression(runs_data, 'green', 'Combined Regression')

plt.xlabel('In Distribution Rank')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


def plot_linear_regression(data, color, label):
    if data:
        x = [run['rankme_rank'] for run in data]
        y = [run['CIFAR100_rank'] + run['caltech101_rank'] for run in data]  # Combine both accuracies
        coeffs = np.polyfit(x, y, 1)
        regression_line = np.poly1d(coeffs)(np.unique(x))
        plt.plot(np.unique(x), regression_line, color=color, linestyle='dashed', label=label)

# Plot scatter for 'vicreg'
for run in vicreg_data:
    plt.scatter(run['rankme_rank'], run['CIFAR100_rank'], color=color_map['vicreg'])
    plt.scatter(run['rankme_rank'], run['caltech101_rank'], color=color_map['vicreg'])

# Plot scatter for 'simclr'
for run in simclr_data:
    plt.scatter(run['rankme_rank'], run['CIFAR100_rank'], color=color_map['simclr'])
    plt.scatter(run['rankme_rank'], run['caltech101_rank'], color=color_map['simclr'])

# Plot linear regressions
#plot_linear_regression(vicreg_data, color_map['vicreg'], 'vicreg Regression')
#plot_linear_regression(simclr_data, color_map['simclr'], 'simclr Regression')
#plot_linear_regression(runs_data, 'green', 'Combined Regression')

plt.xlabel('In Distribution Rank')
plt.ylabel('OOD Rank')
plt.legend()
plt.show()
