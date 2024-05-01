import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv("data/prices_round_2_day_1.csv", sep = ';')

# Extract features and labels
features = data[['TRANSPORT_FEES', 'EXPORT_TARIFF', 'IMPORT_TARIFF', 'SUNLIGHT', 'HUMIDITY']].values
labels = data['ORCHIDS'].values

# Normalize the features
features = (features - features.mean(axis=0)) / features.std(axis=0)

# Convert to PyTorch tensors
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

# Define hyperparameters
input_size = features.shape[1]
hidden_size1 = 64
hidden_size2 = 32
hidden_size3= 16
output_size = 1
learning_rate = 0.001
num_epochs = 80000
step_size = 100
gamma = 0.995

# Define the neural network using Sequential
model = nn.Sequential(
    nn.Linear(input_size, hidden_size1),
    nn.ReLU(),
    nn.Linear(hidden_size1, hidden_size2),
    nn.ReLU(),
    nn.Linear(hidden_size2, hidden_size3),
    nn.ReLU(),
    nn.Linear(hidden_size3, output_size)
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(features)
    loss = criterion(outputs, labels)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    scheduler.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'orchid_price_predictor.pth')

# Print weights
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

sample_row = features[20]
print(data.iloc[20])
sample_row_tensor = torch.tensor(sample_row)

with torch.no_grad():
    predicted_price = model(sample_row_tensor).item()

print(predicted_price)