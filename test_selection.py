import data_utility
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

list_of_clients = [i for i in range(0, 100)]
sampled_clients = np.random.choice(
                list_of_clients,
                size=5,
                replace=False)
print("sampled_clients", sampled_clients)

alpha = 0.3
dataset = "cifar100"

selected_client_examples = data_utility.load_selected_clients_statistics(sampled_clients, alpha, dataset)
print("Local examples selected clients ", selected_client_examples)
print("Total examples ", np.sum(selected_client_examples))
total_examples = np.sum(selected_client_examples)