import torch
from severity_model import SeverityNet

def rule_based_override(features):
    people, vehicles, fire, smoke, sa, sf, sfl, sn = features
    if sn == 1 and people == 0 and fire == 0 and smoke == 0:
        return 0  # LOW
    return None

model = SeverityNet()
model.load_state_dict(torch.load("severity/severity_model.pth"))
model.eval()

sample = [3,2,1,1,1,0,0,0]

override = rule_based_override(sample)

if override is not None:
    severity = override
else:
    pred = model(torch.tensor([sample], dtype=torch.float32))
    severity = pred.argmax(dim=1).item()

labels = ["LOW", "MEDIUM", "HIGH"]
print("Predicted Severity:", labels[severity])
