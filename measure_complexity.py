import sys
import torch
from thop import profile
from thop import clever_format
from absl import flags

from architecture import Model

FLAGS = flags.FLAGS

def measure_model_complexity():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_features = 8
    num_speech_features = 80
    n_phones = 40

    model = Model(num_features, num_speech_features, n_phones).to(device)
    model.eval()

    batch_size = 1
    seq_len = 200

    X = torch.randn(batch_size, seq_len, num_features).to(device)
    X_raw = torch.randn(batch_size, seq_len * 8, num_features).to(device)
    sess = torch.randint(0, 5, (batch_size, seq_len)).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    macs, params_thop = profile(model, inputs=(X, X_raw, sess), verbose=False)
    
    macs_formatted, params_formatted = clever_format([macs, total_params], "%.2f")

    print("\n" + "="*50)
    print("Model Complexity Report")
    print("="*50)
    print(f"🔹 Total Parameters     : {total_params:,} ({params_formatted})")
    print(f"🔹 Trainable Parameters : {trainable_params:,}")
    print(f"🔹 Total MACs           : {int(macs):,} ({macs_formatted})")
    print(f"   * Measured at batch_size={batch_size}, seq_len={seq_len}")
    print("="*50 + "\n")

if __name__ == '__main__':
    FLAGS(sys.argv)
    measure_model_complexity()