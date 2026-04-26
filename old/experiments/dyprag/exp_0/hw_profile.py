"""
Exp 0 — Hardware profiling

Checks whether the base model (Qwen3-8B, 4-bit) + rank-16 LoRA + the chunked
sentence-transformer encoder can fit simultaneously on the available GPU.

This is a prerequisite check before Phase 1 training.
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def gpu_mem_mb():
    """Return (used, total) GPU memory in MB."""
    return (
        torch.cuda.memory_allocated() / 1e6,
        torch.cuda.get_device_properties(0).total_memory / 1e6,
    )


def fmt(used, total):
    return f"{used:.0f} / {total:.0f} MB ({used / total * 100:.1f}%)"


def main():
    if not torch.cuda.is_available():
        print("No CUDA GPU available. Exiting.")
        return

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    total_mb = props.total_memory / 1e6
    print(f"GPU: {props.name}, {total_mb:.0f} MB total")
    print()

    # --- Step 1: Load base model in 4-bit ---
    print("Step 1: Loading Qwen3-8B in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    used, total = gpu_mem_mb()
    base_used = used
    print(f"  After base model load: {fmt(used, total)}")

    # --- Step 2: Add LoRA adapter (rank 16) ---
    print("\nStep 2: Adding rank-16 LoRA adapter...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    used, total = gpu_mem_mb()
    lora_used = used
    print(f"  After LoRA: {fmt(used, total)}")
    print(f"  LoRA overhead: {lora_used - base_used:.0f} MB")

    # --- Step 3: Load sentence-transformer encoder ---
    print("\nStep 3: Loading sentence-transformers/all-MiniLM-L6-v2...")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cuda"
    )
    used, total = gpu_mem_mb()
    encoder_used = used
    print(f"  After encoder: {fmt(used, total)}")
    print(f"  Encoder overhead: {encoder_used - lora_used:.0f} MB")

    # --- Step 4: Test inference ---
    print("\nStep 4: Test inference (forward pass)...")
    test_input = tokenizer(
        "def hello():\n    print('world')\n", return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        output = model(**test_input)
    used, total = gpu_mem_mb()
    print(f"  After forward pass: {fmt(used, total)}")

    # Test encoder
    emb = encoder.encode(["def hello(): print('world')"], convert_to_tensor=True)
    print(f"  Encoder output shape: {emb.shape}")
    used, total = gpu_mem_mb()
    print(f"  After encoder forward: {fmt(used, total)}")

    # --- Summary ---
    free = total - used
    print("\n=== Summary ===")
    print(f"Total GPU memory:     {total:.0f} MB")
    print(f"Used after all loads: {used:.0f} MB")
    print(f"Free:                 {free:.0f} MB")
    print(f"  Base model (4-bit): {base_used:.0f} MB")
    print(f"  LoRA overhead:      {lora_used - base_used:.0f} MB")
    print(f"  Encoder overhead:   {encoder_used - lora_used:.0f} MB")

    fits = free > 1000  # need at least 1GB headroom for training
    print(
        f"\nVerdict: {'FITS' if fits else 'DOES NOT FIT'} "
        f"(~{free:.0f} MB free, need ~1000 MB headroom for training)"
    )

    # Save results
    import json
    from pathlib import Path

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    profile = {
        "gpu": props.name,
        "total_mb": total,
        "base_model_mb": base_used,
        "lora_overhead_mb": lora_used - base_used,
        "encoder_overhead_mb": encoder_used - lora_used,
        "total_used_mb": used,
        "free_mb": free,
        "fits": fits,
    }
    out_path = results_dir / "hw_profile.json"
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\nProfile saved to {out_path}")


if __name__ == "__main__":
    main()
