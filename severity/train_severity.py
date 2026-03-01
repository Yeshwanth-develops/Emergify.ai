import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from severity_model import SeverityNet

# Load data
df = pd.read_csv("datasets/severity/severity_data.csv")

X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
model = SeverityNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(50):
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "severity/severity_model.pth")
print("✅ Severity model trained and saved")
