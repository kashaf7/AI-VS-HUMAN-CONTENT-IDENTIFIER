import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import timm
import tempfile

CKPT_PATH = r"C://Users//syeda//Downloads//user//OneDrive//Desktop//python//AI-VS-HUMAN-CONTENT-DETECTION//backend//main models//epoch_002.ckpt"
FRAMES = 20
IMG = 224
USE_FLOW = True
device = torch.device("cpu")

# -- Model definitions (use your provided classes as-is) --

class XceptionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("xception", pretrained=False, num_classes=0, global_pool="avg")
        self.out_dim = self.model.num_features
    def forward(self, x):
        return self.model(x)
class FrameHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d,1)
    def forward(self, f):
        return self.fc(f)
class SameConv1d(nn.Module):
    def __init__(self, c_in, c_out, k=3, dilation=1):
        super().__init__()
        pad = (dilation*(k-1))//2
        self.conv = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
    def forward(self,x):
        return self.conv(x)
class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dilation=1):
        super().__init__()
        self.c1 = SameConv1d(c_in,c_out,k,dilation)
        self.a1 = nn.ReLU(True)
        self.c2 = SameConv1d(c_out,c_out,1)
        self.a2 = nn.ReLU(True)
        self.proj = nn.Conv1d(c_in,c_out,1) if c_in!=c_out else nn.Identity()
    def forward(self,x):
        y=self.a2(self.c2(self.a1(self.c1(x))))
        return y+self.proj(x)
class TemporalHeadTCN(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(d,hidden,dilation=1),
            TCNBlock(hidden,hidden,dilation=2),
            TCNBlock(hidden,hidden,dilation=4)
        )
        self.fc = nn.Linear(hidden,1)
    def forward(self, seq):
        seq = seq.permute(0,2,1)
        y = self.tcn(seq).mean(-1)
        return self.fc(y)
class FlowCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2,32,5,2,2), nn.ReLU(True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(True),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128,256)
        self.out_dim=256
    def forward(self,x):
        y=self.net(x).flatten(1)
        return self.fc(y)
class MultiStreamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = XceptionBackbone()
        D = self.backbone.out_dim
        self.frame_head = FrameHead(D)
        self.video_tcn = TemporalHeadTCN(D, 512)
        self.use_flow = USE_FLOW
        if USE_FLOW:
            self.flow_cnn = FlowCNN()
            self.flow_tcn = TemporalHeadTCN(self.flow_cnn.out_dim, 256)
    def forward(self, clip, flow=None):
        B,T,C,H,W = clip.shape
        feat = self.backbone(clip.view(B*T,C,H,W))
        D=feat.size(-1)
        seq=feat.view(B,T,D)
        f_frame=self.frame_head(feat).view(B,T,1).mean(1)
        f_video=self.video_tcn(seq)
        fused=0.5*f_video + 0.3*f_frame
        if self.use_flow and flow is not None and flow.numel()>0:
            B,Tf,_,_,_ = flow.shape
            f=self.flow_cnn(flow.view(B*Tf,2,IMG,IMG))
            seq_f=f.view(B,Tf,-1)
            f_flow=self.flow_tcn(seq_f)
            fused = fused + 0.2*f_flow
        return fused

# -- Load model as singleton on import --
model = MultiStreamModel().to(device)
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def read_frames_from_bytes(video_bytes, k):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as fp:
        fp.write(video_bytes.read() if hasattr(video_bytes, 'read') else video_bytes)
        temp_path = fp.name

    cap = cv2.VideoCapture(temp_path)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx=np.linspace(0,total-1,k).astype(int)
    frames=[]
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok: continue
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (IMG,IMG))
        frames.append(f)
    cap.release()
    Path(temp_path).unlink(missing_ok=True)
    while len(frames) < k: frames.append(frames[-1])
    return np.stack(frames)

def predict_video_bytes(video_bytes):
    frames = read_frames_from_bytes(video_bytes, FRAMES)
    clip = torch.stack([tfm(x) for x in frames]).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(clip, None)
        prob = torch.sigmoid(logit).item()
    label = "FAKE (AI)" if prob > 0.5 else "REAL (Human)"
    return {"prob_ai": prob, "prob_human": 1.0-prob, "label": label}