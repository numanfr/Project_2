
import os
import numpy as np
import torch
import json

dataset_folder = '../data/'
files = os.listdir(dataset_folder)
files = [f for f in files if f.endswith('.npz')]

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


preferred_parameters = {
    'FashionMNIST0.3.npz': {
        'num_epochs': 10,
        'batch_size': 8192,
        'learning_rate': 0.001,
        'weight_decay': 1e-4
    },
    'FashionMNIST0.6.npz': {
        'num_epochs': 5,
        'batch_size': 8192,
        'learning_rate': 0.001,
        'weight_decay': 1e-4
    },
    'CIFAR.npz': {
        'num_epochs': 50,
        'batch_size': 8192,
        'learning_rate': 0.00001,
        'weight_decay': 1e-4
    }
}


def save_results(results, id):
    with open(f"../results/{id}.json", 'w') as f:
        json.dump(results, f, indent=4)

def run_single_experiment(validation_ratio = 0.2, predict_transition_matrix = False, id = 0):
    
    results = {}
    
    for file in files:
        dataset = np.load(os.path.join(dataset_folder, file))
        print(f"File: {file}")
        Xtr = dataset['Xtr']
        Str = dataset['Str']
        Xts = dataset['Xts']
        Yts = dataset['Yts']

        use_transition_matrix = True  # Set to False to ignore the transition matrix
        if file == 'CIFAR.npz':
            use_transition_matrix = False
            Xtr = Xtr.reshape(Xtr.shape[0], -1)
            Xts = Xts.reshape(Xts.shape[0], -1)

        if predict_transition_matrix:
            use_transition_matrix = False

        Xtr = Xtr / 255.0
        Xts = Xts / 255.0



        n = Xtr.shape[0]
        n_val = int(n * validation_ratio)
        indices = np.random.permutation(n)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train_tensor = torch.tensor(Xtr[train_indices], dtype=torch.float32)
        S_train_tensor = torch.tensor(Str[train_indices], dtype=torch.long)
        X_val_tensor = torch.tensor(Xtr[val_indices], dtype=torch.float32)
        S_val_tensor = torch.tensor(Str[val_indices], dtype=torch.long)


        #one-hot encoding

        S_train_tensor = torch.nn.functional.one_hot(S_train_tensor, num_classes=3).float()
        S_val_tensor = torch.nn.functional.one_hot(S_val_tensor, num_classes=3).float()


        if use_transition_matrix:
            transition_matrix = transition_matrices[file]
            transition_matrix_tensor = torch.tensor(transition_matrix, dtype=torch.float32, requires_grad=False)
            
            

        # layer_sizes = [32, 64, 32]
        layer_sizes = [256, 128, 64]
        layers = []
        input_size = Xtr.shape[1]
        for size in layer_sizes:
            layers.append(torch.nn.Linear(input_size, size))
            layers.append(torch.nn.LayerNorm(size))
            layers.append(torch.nn.ReLU())
            input_size = size
        layers.append(torch.nn.Linear(input_size, 3))
        layers.append(torch.nn.Softmax(dim=1))
        model = torch.nn.Sequential(*layers)

        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=preferred_parameters[file]['learning_rate'], weight_decay=preferred_parameters[file]['weight_decay'])
        num_epochs = preferred_parameters[file]['num_epochs']
        batch_size = preferred_parameters[file]['batch_size']
        train_losses = []
        val_losses = []
        val_accuracies = []
        test_accuracies = []
        for epoch in range(num_epochs):
            model.train()
            permutation = torch.randperm(X_train_tensor.shape[0])
            for i in range(0, X_train_tensor.shape[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_s = X_train_tensor[indices], S_train_tensor[indices]

                optimizer.zero_grad()
                outputs = model(batch_x)
                if use_transition_matrix:
                    outputs = torch.matmul(outputs, transition_matrix_tensor)
                loss = criterion(outputs, batch_s)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                if use_transition_matrix:
                    val_outputs = torch.matmul(val_outputs, transition_matrix_tensor)
                val_loss = criterion(val_outputs, S_val_tensor)
                _, predicted = torch.max(val_outputs, 1)
                accuracy = (predicted == S_val_tensor.argmax(dim=1)).float().mean()
                
                test_outputs = model(torch.tensor(Xts, dtype=torch.float32))
                _, test_predicted = torch.max(test_outputs, 1)
                test_accuracy = (test_predicted.numpy() == Yts).mean()
                test_accuracies.append(test_accuracy)
                
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            val_accuracies.append(accuracy.item())
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {accuracy.item():.4f}, Test Acc: {test_accuracy:.4f}")

        results[file] = test_accuracies[-1]
        
        if predict_transition_matrix:
            
            model.eval()
            with torch.no_grad():
                y_pred = model(X_train_tensor)
                _, y_pred_classes = torch.max(y_pred, 1)

            y_pred = y_pred.numpy()
            y_pred_classes = y_pred_classes.numpy()
            matrix = np.zeros((3, 3))
            for column in range(3):

                highest_confidence_pred_index = y_pred[y_pred_classes == column][:, column].argmax()
                highest_confidence_pred = y_pred[y_pred_classes == column][highest_confidence_pred_index]
                # print(highest_confidence_pred_index, highest_confidence_pred)
                matrix[column] = highest_confidence_pred

            results[file]['predicted_transition_matrix'] = matrix

    return results
