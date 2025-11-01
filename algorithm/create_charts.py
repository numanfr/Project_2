from matplotlib import pyplot as plt
import json
import numpy as np
import os

results_dir = "results/"
chart_dir = "charts/"


transition_matrices = {
    'FashionMNIST0.3.npz': np.array([
        [0.7, 0.3, 0.0],
        [0.0, 0.7, 0.3],
        [0.3, 0.0, 0.7]
    ]),
    'FashionMNIST0.6.npz': np.array([
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4]
    ]),
}


def plot_bar_charts():
    for dataset in test_accuracies:
        
        #plot bar charts for test accuracies
        algorithms = list(test_accuracies[dataset].keys())
        accuracies = [test_accuracies[dataset][algo] for algo in algorithms]
        stds = [test_stds[dataset][algo] for algo in algorithms]
        x = np.arange(len(algorithms))
        plt.figure(figsize=(10, 6))
        plt.bar(x, accuracies, yerr=stds, capsize=5)
        plt.xticks(x, algorithms, rotation=45)
        plt.ylabel('Test Accuracy')
        plt.title(f'Test Accuracies for {dataset}')
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, f"{dataset}_test_accuracies.png"))
        plt.show()



def print_transition_matrices():
    for dataset in test_t_matrices:
        print(f"Dataset: {dataset}")
        is_cifar = dataset not in transition_matrices

        if is_cifar:
            fig, axs = plt.subplots(1, 3, figsize=(18, 7))
            axs = axs.flatten()
        else:
            fig, axs = plt.subplots(1, 4, figsize=(18, 6))
            axs = axs.flatten()

        if not is_cifar:
            gt_matrix = transition_matrices[dataset]
            im = axs[0].imshow(gt_matrix, cmap='viridis', vmin=0, vmax=1)
            axs[0].set_title('Ground Truth Transition Matrix')
            for i in range(gt_matrix.shape[0]):
                for j in range(gt_matrix.shape[1]):
                    axs[0].text(j, i, f"{gt_matrix[i, j]:.2f}", ha='center', va='center', color='w')
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            # fig.colorbar(im, ax=axs[0])
            start_idx = 1
        else:
            start_idx = 0
        for idx, algo in enumerate(test_t_matrices[dataset]):
            matrix = test_t_matrices[dataset][algo]
            # print(f"Algorithm: {algo}")
            # print(matrix)
            im = axs[start_idx + idx].imshow(matrix, cmap='viridis', vmin=0, vmax=1)
            axs[start_idx + idx].set_title(f'Predicted by {algo}')
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axs[start_idx + idx].text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', color='w')
            axs[start_idx + idx].set_xticks([])
            axs[start_idx + idx].set_yticks([])
            # fig.colorbar(im, ax=axs[start_idx + idx])
        plt.suptitle(f'Transition Matrices for {dataset}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, f"{dataset}_transition_matrices.png"))
        plt.show()



def create_charts():

    os.makedirs(chart_dir, exist_ok=True)

    files = os.listdir(results_dir)
    files = [f for f in files if f.endswith('.json')]

    results = {}
    for file in files:
        algorithm = file.split('_')[1].split('.')[0]
        with open(os.path.join(results_dir, file), 'r') as f:
            data = json.load(f)
        results[algorithm] = data

    test_accuracies = {}
    test_stds = {}
    test_t_matrices = {}
    test_t_matrices_stds = {}



    for algo in results:
        for dataset in results[algo]:
            if dataset not in test_accuracies:
                test_accuracies[dataset] = {}
                test_stds[dataset] = {}
                test_t_matrices[dataset] = {}
                test_t_matrices_stds[dataset] = {}

    for algo in results:
        for dataset in results[algo]:
            test_accuracies[dataset][algo] = results[algo][dataset]['test_accuracy']['mean']
            test_stds[dataset][algo] = results[algo][dataset]['test_accuracy']['std']
            try:
                test_t_matrices[dataset][algo] = np.array(results[algo][dataset]['predicted_transition_matrix']['mean'])
                test_t_matrices_stds[dataset][algo] = np.array(results[algo][dataset]['predicted_transition_matrix']['std'])
            except KeyError:
                continue

    plot_bar_charts()
    print_transition_matrices()

if __name__ == "__main__":
    create_charts()