import torch

# Configuration des chemins et tailles
input_path = 'dataset.pt'
train_output = 'train.pt'
val_output = 'val.pt'
test_output = 'test.pt'
train_size = 800
val_size = 100
test_size = 100
seed = 42

# Chargement du dataset
data = torch.load(input_path)
if not isinstance(data, dict) or 'xtrue' not in data or 'yblur' not in data:
    raise ValueError("Le fichier doit contenir un dict avec les clés 'xtrue' et 'yblur'.")

x = data['xtrue'].T
y = data['yblur'].T

print(x.shape, y.shape)

# Mélange des indices
torch.manual_seed(seed)
indices = torch.randperm(1000)

train_idx = indices[:train_size]
val_idx = indices[train_size:train_size + val_size]
test_idx = indices[train_size + val_size:train_size + val_size + test_size]

# Création des splits
train_data = {'xtrue': x[train_idx].T, 'yblur': y[train_idx].T}
val_data   = {'xtrue': x[val_idx].T,   'yblur': y[val_idx].T}
test_data  = {'xtrue': x[test_idx].T,  'yblur': y[test_idx].T}

# Sauvegarde
torch.save(train_data, train_output)
torch.save(val_data,   val_output)
torch.save(test_data,  test_output)

print(f"Saved {train_size} samples to '{train_output}' and {val_size} samples to '{val_output}' and {test_size} samples to '{test_output}'.")
