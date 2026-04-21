import io, base64
from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# ── Model definition ─────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, residual=False):
        super().__init__()
        self.residual = residual and (in_ch == out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        out = self.block(x)
        return out + x if self.residual else out

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))

class EuroSATNet(nn.Module):
    def __init__(self, num_classes=10, drop=0.4):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3,   64),           # 0
            ConvBlock(64,  64,  True),    # 1
            nn.MaxPool2d(2),              # 2
            ConvBlock(64,  128),          # 3
            ConvBlock(128, 128, True),    # 4
            nn.MaxPool2d(2),              # 5
            ConvBlock(128, 256),          # 6
            ConvBlock(256, 256, True),    # 7
            ConvBlock(256, 256, True),    # 8
            SpatialAttention(),           # 9
            nn.MaxPool2d(2),              # 10
            ConvBlock(256, 512),          # 11
            ConvBlock(512, 512, True),    # 12
            SpatialAttention(),           # 13
            nn.AdaptiveAvgPool2d(1),      # 14
            nn.Flatten(),                 # 15
            nn.GELU(),                    # 16
            nn.Linear(512, 256),          # 17
            nn.GELU(),                    # 18
            nn.Dropout(drop),             # 19
            nn.Linear(256, num_classes),  # 20
        )
    def forward(self, x):
        return self.net(x)

# ── Load checkpoint ──────────────────────────────────────────────────────────
CKPT = torch.load(
    '/mnt/user-data/uploads/eurosat_cnn_complete.pth',
    map_location='cpu', weights_only=False
)
CLASS_NAMES = CKPT['class_names']
IMG_SIZE    = CKPT['img_size']
MEAN        = CKPT['mean']
STD         = CKPT['std']
VAL_ACC     = CKPT['val_acc']

model = EuroSATNet(num_classes=len(CLASS_NAMES))
model.load_state_dict(CKPT['model_state_dict'])
model.eval()

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']
    img = Image.open(img_file.stream).convert('RGB')

    # Inference
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top5_idx = probs.argsort(descending=True)[:5].tolist()
    top5 = [
        {'label': CLASS_NAMES[i], 'prob': round(probs[i].item() * 100, 2)}
        for i in top5_idx
    ]

    return jsonify({
        'top_label':   CLASS_NAMES[top5_idx[0]],
        'top_prob':    round(probs[top5_idx[0]].item() * 100, 2),
        'top5':        top5,
        'all_probs':   [round(p * 100, 2) for p in probs.tolist()],
        'class_names': CLASS_NAMES,
    })

@app.route('/info')
def info():
    return jsonify({
        'class_names': CLASS_NAMES,
        'val_acc': round(VAL_ACC * 100, 2),
        'img_size': IMG_SIZE,
    })

if __name__ == '__main__':
    print("🛰️  EuroSAT Classifier running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
