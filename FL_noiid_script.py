
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
import random
import os
import time
import pickle


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras

# Load the attack_types list
with open(f'attack_types.pkl', 'rb') as f:
    attack_types = pickle.load(f)

    
clients = []
num_clients = 7

# Ask for the folder number
folder_number = input("Enter the folder number (1 or 2): ")

# Load the client data from the selected folder
client_data = []
for i in range(num_clients):
    client_data.append(pd.read_csv(f'FL_non_IID_{folder_number}/client{i+1}_data.csv'))


client_x = []
client_y = []
client_index = []
attack_type_list = []


for i in client_data:
    # Generate attack_type dictionary for each client
    attack_type_dict = i['attack_type'].to_dict()
    index = i.index
    x = i.drop(['outcome', 'level', "attack_type"], axis=1).values
    y = i[["outcome"]].values
    y_reg = i['level'].values
    index_array = index.values
    
    client_x.append(x)
    client_y.append(y)
    client_index.append(index_array)
    attack_type_list.append(attack_type_dict)


# Split x, y, and index_array into train and test
x_train, x_test, y_train, y_test, index_train, index_test, attack_type_train_list, attack_type_test_list = [], [], [], [], [], [], [], []

for i in range(num_clients):
    x_train_local, x_test_local, y_train_local, y_test_local, index_train_local, index_test_local = train_test_split(client_x[i], client_y[i], client_index[i], test_size=0.2, random_state=42)
    attack_type_dict_train = {index: str(attack_type_list[i][index]) for index in index_train_local}
    attack_type_dict_test = {index: str(attack_type_list[i][index]) for index in index_test_local}
    x_train.append(x_train_local)
    x_test.append(x_test_local)
    y_train.append(y_train_local)
    y_test.append(y_test_local)
    index_train.append(index_train_local)
    index_test.append(index_test_local)
    attack_type_train_list.append(attack_type_dict_train)
    attack_type_test_list.append(attack_type_dict_test) 



# Create a list to store all clients' x_train, y_train, x_test, and y_test
local_datasets = []

# Iterate over each client
for i in range(num_clients):
    # Get the data for the current client
    client_x_train = x_train[i]
    client_y_train = y_train[i]
    client_x_test = x_test[i]
    client_y_test = y_test[i]
    
    # Append the data to the list
    local_datasets.append((client_x_train, client_y_train, client_x_test, client_y_test))



global_model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(local_datasets[0][0].shape[1:]), 
                          kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), 
                          bias_regularizer=regularizers.L2(1e-4),
                          activity_regularizer=regularizers.L2(1e-5)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=128, activation='relu', 
                          kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), 
                          bias_regularizer=regularizers.L2(1e-4),
                          activity_regularizer=regularizers.L2(1e-5)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=512, activation='relu', 
                          kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), 
                          bias_regularizer=regularizers.L2(1e-4),
                          activity_regularizer=regularizers.L2(1e-5)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=128, activation='relu', 
                          kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), 
                          bias_regularizer=regularizers.L2(1e-4),
                          activity_regularizer=regularizers.L2(1e-5)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=1, activation='sigmoid'),
])

# Create a new directory for this session
session_dir = f"session_{int(time.time())}"
os.makedirs(session_dir, exist_ok=True)

os.makedirs(f'{session_dir}/evaluation', exist_ok=True)
os.makedirs(f'{session_dir}/models', exist_ok=True)
os.makedirs(f'{session_dir}/models/global', exist_ok=True)
for i in range(7):
    os.makedirs(f'{session_dir}/models/client{i}', exist_ok=True)

print(f"New session directory created: {session_dir}")




# Now replace all instances of 'FL_non_IID_2/DoS/models' with '{session_dir}/models'
global_model.save(f'{session_dir}/models/global/model_0.h5')

# local loss and accuracy list for each client
local_loss, local_acc = {}, {}
global_loss, global_acc = {}, {}
clients_this_round = {}

# Ask for the number of global rounds
num_rounds = int(input("Enter the number of global rounds: "))

# Ask for the clients that will not train in each round
clients_not_training = input("Enter the clients that will not train in each round (format: round:client,client,...; round:client,client,...; ...): ")
clients_not_training = {int(round_clients.split(':')[0]): [int(client) for client in round_clients.split(':')[1].split(',')] for round_clients in clients_not_training.split(';')}

for i in range(7):
    local_loss[i] = []
    local_acc[i] = []

for i in range(num_rounds):
    clients_this_round[i] = []
    global_loss[i] = []
    global_acc[i] = []

for round in range(num_rounds):
    # Train the models locally on each client
    print(f'Round {round}')
    clients_this_round = ()
    for i, (client_x_train, client_y_train, client_x_test, client_y_test) in enumerate(local_datasets):
        if round in clients_not_training and i in clients_not_training[round]:
            continue

        clients_this_round += (i,)
        # Load the global model
        model = tf.keras.models.load_model(f'{session_dir}/models/global/model_{round}.h5')



        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
        x_train = client_x_train.astype(np.float32)
        x_test = client_x_test.astype(np.float32)
        y_train = client_y_train.astype(np.float32)
        y_test = client_y_test.astype(np.float32)

        # Train the model locally
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, verbose=1, batch_size= 32)

        # Save accuracy and loss for this model and round
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        local_loss[i].append(loss)
        local_acc[i].append(acc)
        # Save the locally trained model
        model.save(f'{session_dir}/models/client{i}/model_round_{round}.h5')


    # Load the locally trained models
    models = []
    print(f'Clients this round: {clients_this_round}')
    for i in clients_this_round:
        print(f'Loading model for client {i}')
        models.append(tf.keras.models.load_model(f'{session_dir}/models/client{i}/model_round_{round}.h5'))


    # Check if any models were loaded
    if not models:
        print(f'No models were loaded in round {round}, skipping this round.')
        continue

    # Get the number of layers
    num_layers = len(models[0].get_weights())

    # Calculate the average weights for each layer
    average_weights = [np.mean([model.get_weights()[i] for model in models], axis=0) for i in range(num_layers)]
    # Load the global model
    global_model = tf.keras.models.load_model(f'{session_dir}/models/global/model_{round}.h5')


    # Set the weights of the global model to the average weights
    global_model.set_weights(average_weights)

    # Compile the global model
    global_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    global_loss[round].append(loss)
    global_acc[round].append(acc)

    # Save the updated global model
    global_model.save(f'{session_dir}/models/global/model_{round+1}.h5')









# Create a new figure with 2 subplots for local metrics: one for loss, one for accuracy
fig, axs = plt.subplots(2, figsize=(10, 10))

# Plot the local loss scores on the first subplot
for client in local_loss.keys():
    axs[0].plot(range(1, len(local_loss[client]) + 1), local_loss[client], marker='o', linestyle='-', label=f'Client {client} Local Loss')

# Add a legend
axs[0].legend(loc='best')

# Add labels and title
axs[0].set_xlabel('Round')
axs[0].set_ylabel('Loss')
axs[0].set_title('Local Loss Scores for Each Client')

# Plot the local accuracy scores on the second subplot
for client in local_acc.keys():
    axs[1].plot(range(1, len(local_acc[client]) + 1), local_acc[client], marker='o', linestyle='-', label=f'Client {client} Local Accuracy')

# Add a legend
axs[1].legend(loc='best')

# Add labels and title
axs[1].set_xlabel('Round')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Local Accuracy Scores for Each Client')

# Show the plot
plt.tight_layout()
plt.savefig(f'{session_dir}/local_metrics.png')
plt.close()










print(f"clients not training: {clients_not_training}")
# Create a DataFrame with a binary indicator for each client and round
df = pd.DataFrame(0, index=range(0, num_rounds), columns=range(0, 7))

for round_number, clients in clients_not_training.items():
    # Set the value to 1 for clients that are not training
    df.loc[round_number, clients] = 1

# Invert the DataFrame to get a binary indicator of the clients that are training
df = 1 - df

# Plot the heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(df, cmap="Greys_r", cbar=False, linewidths=.5)

# Add labels and title
plt.xlabel('Client')
plt.ylabel('Round')
plt.title('Training Participation by Client and Round')

# Show the plot
plt.savefig(f'{session_dir}/training_participation_heatmap.png')







# Split x, y, and index_array into train and test
x_train, x_test, y_train, y_test, index_train, index_test, attack_type_train_list, attack_type_test_list = [], [], [], [], [], [], [], []

for i in range(num_clients):
    x_train_local, x_test_local, y_train_local, y_test_local, index_train_local, index_test_local = train_test_split(client_x[i], client_y[i], client_index[i], test_size=0.2, random_state=42)
    attack_type_dict_train = {index: str(attack_type_list[i][index]) for index in index_train_local}
    attack_type_dict_test = {index: str(attack_type_list[i][index]) for index in index_test_local}
    x_train.append(x_train_local)
    x_test.append(x_test_local)
    y_train.append(y_train_local)
    y_test.append(y_test_local)
    index_train.append(index_train_local)
    index_test.append(index_test_local)
    attack_type_train_list.append(attack_type_dict_train)
    attack_type_test_list.append(attack_type_dict_test) 





#Load last global model
model = tf.keras.models.load_model(f'{session_dir}/models/global/model_{(num_rounds)}.h5')


# Loop over all clients
for i in range(len(x_test)):
    # Get the original data for this client
    x_client = np.array(x_test[i], dtype='float32')
    y_client = y_test[i]

    # Reshape your data if necessary
    x_client = x_client.reshape((x_client.shape[0], -1))

    # Predict the labels for the data
    y_pred = model.predict(x_client)

    # Round the probabilities to get the predicted classes
    y_pred = np.round(y_pred).astype(int)

    # Convert y_pred to a DataFrame
    y_pred_df = pd.DataFrame(y_pred, columns=['Predicted Label'])

    # Add a column with the true attack type of each sample
    y_pred_df['True Attack Type'] = [attack_type_test_list[i][idx] for idx in index_test[i]]

    # Add a column with the true label
    y_pred_df['True Label'] = y_client

    # Export y_pred_df as a CSV file
    y_pred_df.to_csv(f'{session_dir}/evaluation/predictions_client_{i}.csv', index=False)





# For each client
for i in range(len(x_test)):
    # Read the predictions DataFrame for this client
    y_pred_df = pd.read_csv(f'{session_dir}/evaluation/predictions_client_{i}.csv')

    # Get the predicted labels and true labels
    y_pred = y_pred_df['Predicted Label']
    y_true = y_pred_df['True Label']

    # Initialize a DataFrame to store the results
    results_df = pd.DataFrame(columns=['Attack Type', 'TP', 'FP', 'TN', 'FN'])

    # For each attack type
    for attack_type in y_pred_df['True Attack Type'].unique():
        # Compute the TP, FP, TN, FN
        tp = np.sum((y_pred == 1) & (y_true == 1) & (y_pred_df['True Attack Type'] == attack_type))
        fp = np.sum((y_pred == 1) & (y_true == 0) & (y_pred_df['True Attack Type'] == attack_type))
        tn = np.sum((y_pred == 0) & (y_true == 0) & (y_pred_df['True Attack Type'] == attack_type))
        fn = np.sum((y_pred == 0) & (y_true == 1) & (y_pred_df['True Attack Type'] == attack_type))

        # Add the results to the DataFrame
        temp_df = pd.DataFrame({'Attack Type': [attack_type], 'TP': [tp], 'FP': [fp], 'TN': [tn], 'FN': [fn]})
        temp_df[['TP', 'FP', 'TN', 'FN']] = temp_df[['TP', 'FP', 'TN', 'FN']].astype(int)
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
        #save the results to a CSV file
        results_df.to_csv(f'{session_dir}/evaluation/results_client_{i}.csv', index=False)

    # Print the results
    print(f"Client {i+1} results:")
    print(results_df)


# For each client
for i in range(len(x_test)):
    # Read the results DataFrame for this client
    results_df = pd.read_csv(f'{session_dir}/evaluation/results_client_{i}.csv')

    # Convert the TP, FP, TN, and FN columns to integers
    results_df[['TP', 'FP', 'TN', 'FN']] = results_df[['TP', 'FP', 'TN', 'FN']].astype(int)

    # Print the results
    print(f"Client {i+1} results:")
    print(results_df)

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(results_df.set_index('Attack Type'), annot=True, fmt='d', cmap='viridis')
    plt.title(f'Client {i+1} results')
    plt.savefig(f'{session_dir}/evaluation/heatmap_client_{i+1}.png')



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

all_results_df = pd.DataFrame()
# For each client
for i in range(7):
    # Read the results DataFrame for this client
    results_df = pd.read_csv(f'{session_dir}/evaluation/results_client_{i}.csv')
    # Concatenate this dataframe to the all_results_df. If attack type is the same, the results will be added together
    all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
    
all_results_df = all_results_df.groupby('Attack Type').sum().reset_index()
all_results_df.to_csv(f"{session_dir}/evaluation/all.csv")

all_results_df[["TP", "FP", "TN", "FN"]] = all_results_df[["TP", "FP", "TN", "FN"]].astype(int)

# Calculate recall, precision, and F1 score
all_results_df['Recall'] = all_results_df['TP'] / (all_results_df['TP'] + all_results_df['FN'])
all_results_df['Precision'] = all_results_df['TP'] / (all_results_df['TP'] + all_results_df['FP'])
all_results_df['F1 Score'] = 2 * (all_results_df['Precision'] * all_results_df['Recall']) / (all_results_df['Precision'] + all_results_df['Recall'])

# Calculate total
all_results_df['Total'] = all_results_df[['TP', 'FP', 'TN', 'FN']].sum(axis=1)

# Add a row for TP, FP, TN and FN total
totals = all_results_df[['TP', 'FP', 'TN', 'FN', 'Recall', 'Precision', 'F1 Score', 'Total']].sum(axis=0)
totals.name = 'Totals'
all_results_df = pd.concat([all_results_df, pd.DataFrame(totals).T])

plt.figure(figsize=(10, 8))
sns.heatmap(all_results_df.set_index('Attack Type'), annot=True, fmt='.2f', cmap='viridis')
plt.title('All clients results')
plt.savefig(f'{session_dir}/evaluation/heatmap_all.png')


print(f"Training and evaluation finished, remember you can find all docs in: {session_dir}")