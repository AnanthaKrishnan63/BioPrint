"""Export only the trained encoder for optional production use, no data reads."""
import hashlib
import json
import pathlib
import torch
from pointer_sapimouse_benchmark import FCN

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/benchmarks/pointer_sapimouse'

class EncoderOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model.embed(x)

if __name__ == '__main__':
    torch.set_num_threads(2)
    if not (OUT / 'results.json').exists():
        raise SystemExit('Wait for the frozen benchmark to finish before exporting its final checkpoint')
    saved = torch.load(OUT / 'fcn_training_best.pt', weights_only=True)
    model = FCN(saved['classes']).eval()
    model.load_state_dict(saved['state_dict'])
    wrapper = EncoderOnly(model).eval()
    example = torch.randn(2, 2, 128)
    traced = torch.jit.trace(wrapper, example)
    with torch.inference_mode():
        torch.testing.assert_close(traced(example), wrapper(example))
    target = OUT / 'encoder.torchscript.pt'
    traced.save(str(target))
    manifest = {'architecture': 'Conv1d128k8-BN-ReLU-Conv1d256k5-BN-ReLU-Conv1d128k3-BN-ReLU-GAP', 'input_shape': ['blocks', 2, 128], 'output_features': 128, 'normalization': 'absolute xy first-differences; per-block z-score both channels', 'training_checkpoint_epoch': saved['epoch'], 'torch_version': str(torch.__version__), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'production_default_enabled': False}
    (OUT / 'encoder_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
