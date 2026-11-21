"""Build a separate standard W8A8 feasibility checkpoint, without calibration data.

Not a claimed competition innovation. Original BF16 weights remain read-only.
"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPECTED_SHA = "14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/workspace/MiniCPM5-2B")
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    source, dest = Path(args.source), Path(args.destination)
    if dest.exists():
        raise FileExistsError(f"preserving existing candidate {dest}")
    src_weight = source / "model-00000-of-00001.safetensors"
    sha = hashlib.file_digest(src_weight.open("rb"), "sha256").hexdigest()
    if sha != EXPECTED_SHA:
        raise AssertionError("source checkpoint differs from verified original")
    cfg = json.loads((source / "config.json").read_text())
    quant = {"quant_method": "compressed-tensors", "format": "int-quantized",
             "quantization_status": "compressed", "ignore": ["lm_head"],
             "config_groups": {"group_0": {"targets": ["Linear"],
                 "weights": {"num_bits": 8, "type": "int", "symmetric": True,
                             "strategy": "channel", "dynamic": False},
                 "input_activations": {"num_bits": 8, "type": "int", "symmetric": True,
                                       "strategy": "token", "dynamic": True},
                 "output_activations": None}}, "kv_cache_scheme": None}
    from compressed_tensors.quantization import QuantizationConfig
    QuantizationConfig.model_validate(quant)
    dest.mkdir(parents=True)
    torch.set_num_threads(4)
    tensors, audit = {}, []
    torch.cuda.set_device(0)
    with safe_open(str(src_weight), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            weight = handle.get_tensor(key)
            selected = key.startswith("model.layers.") and key.endswith(".weight") and weight.ndim == 2
            if selected:
                w = weight.cuda().float()
                maxabs = w.abs().amax(dim=1, keepdim=True)
                scale = torch.where(maxabs > 0, maxabs / 127, torch.ones_like(maxabs))
                q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
                reconstructed = q.float() * scale
                audit.append({"name": key, "shape": list(weight.shape),
                              "relative_l2": float((reconstructed-w).norm()/w.norm().clamp_min(1e-30)),
                              "max_abs_error": float((reconstructed-w).abs().max())})
                tensors[key] = q.cpu().contiguous()
                tensors[key.removesuffix(".weight") + ".weight_scale"] = scale.cpu().contiguous()
                del w, q, scale, reconstructed
            else:
                tensors[key] = weight.contiguous()
    if len(audit) != cfg["num_hidden_layers"] * 7:
        raise AssertionError(f"unexpected number of quantized projections: {len(audit)}")
    target_weight = dest / "model.safetensors"
    save_file(tensors, str(target_weight), metadata={"format": "pt"})
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and not path.name.endswith(".index.json") and path.name != "config.json":
            shutil.copy2(path, dest / path.name)
    cfg["quantization_config"] = quant
    (dest / "config.json").write_text(json.dumps(cfg, indent=2))
    report = {"created": datetime.now(timezone.utc).isoformat(), "kind": "standard_W8A8_feasibility_only",
              "original_sha256": sha, "candidate_sha256": hashlib.file_digest(target_weight.open("rb"), "sha256").hexdigest(),
              "quantized_tensors": len(audit), "weights": audit,
              "compressed_tensors_version": importlib.metadata.version("compressed-tensors"),
              "model_accuracy_proven": False, "calibration_or_evaluation_data_used": False}
    (dest / "conversion_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "weights"}), flush=True)


if __name__ == "__main__":
    main()

