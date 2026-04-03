import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import timm

MODEL_PATH = r"C://Users//syeda//Downloads//user//OneDrive//Desktop//python//AI-VS-HUMAN-CONTENT-DETECTION//backend//main models//xcheckpoint_epoch_21.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 299
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        new_state = {}
        for k,v in state_dict.items():
            new_key = k[len("module."):] if k.startswith("module.") else k
            new_state[new_key] = v
        return new_state
    return state_dict

def extract_state_dict_from_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                sd = ckpt[key]
                return strip_module_prefix(sd)
        return strip_module_prefix(ckpt)
    else:
        return ckpt

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

model = timm.create_model("xception", pretrained=False, num_classes=1)

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

ckpt = torch.load(MODEL_PATH, map_location="cpu")
state_dict = extract_state_dict_from_checkpoint(ckpt)

try:
    model.load_state_dict(state_dict, strict=True)
except RuntimeError as e:
    print("Strict load failed:", e)
    print("Attempting partial load (strict=False) and skipping mismatched keys...")
    model.load_state_dict(state_dict, strict=False)

model = model.to(DEVICE)
model.eval()

def predict_image_bytes(image_bytes):
    try:
        im = Image.open(image_bytes).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Cannot identify image: {e}")
    x = transform(im).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.view(-1)
        logit = out[0].cpu().item()
        prob_ai = float(torch.sigmoid(torch.tensor(logit)).item())
        prob_human = 1.0 - prob_ai
    label = "FAKE (AI)" if prob_ai >= 0.5 else "REAL (Human)"
    return {
        "prob_ai": prob_ai,
        "prob_human": prob_human,
        "label": label
    }