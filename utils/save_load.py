import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_name, path):
    if model_name == 'scGPT':
        from models.scgpt_model import SCGPTModel
        model = SCGPTModel()
    elif model_name == 'HyenaDNA':
        from models.hyenadna_model import HyenaDNAModel
        model = HyenaDNAModel()
    elif model_name == 'MAMBA':
        from models.mamba_model import MambaModel
        model = MambaModel()
    else:
        raise ValueError(f"Unknown model name {model_name}")

    model.load_state_dict(torch.load(path))
    model.eval()
    return model
