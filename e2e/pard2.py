"""Generate text with the paper's GPTQ W4A4KV4 target and optional PARD-2 drafter."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e.checkpoint_utils.require_gptq import require_gptq_checkpoint
from e2e.speculative import PARD2_PROFILES, Pard2Spec, load_runtime


def verify_local_resources(draft, benchmark_profile, mode):
    """Check the pinned drafter files needed by the selected inference mode."""
    required = {"config.json": 1, "model.safetensors": 1_000_000_000}
    if mode == "pard2-td":
        required["warp_model.bin"] = 1_000_000
    failures = [str(Path(draft) / name) for name, minimum in required.items()
                if not (Path(draft) / name).is_file()
                or (Path(draft) / name).stat().st_size < minimum]
    if failures:
        spec = Pard2Spec.for_benchmark_profile(benchmark_profile, mode=mode)
        raise FileNotFoundError(
            "The pinned PARD-2 snapshot is incomplete. Download it with:\n"
            f"hf download {spec.model_id} --revision {spec.revision}\n"
            + "\n".join(failures))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("ar", "pard2-ti", "pard2-td"), required=True)
    result.add_argument("--target", required=True, help="Local converted GPTQ checkpoint.")
    result.add_argument("--tokenizer", required=True, help="Local pinned Qwen3 snapshot.")
    result.add_argument("--draft", help="Local pinned PARD-2 snapshot; required for TI/TD.")
    result.add_argument("--profile", "--benchmark-profile", dest="benchmark_profile",
                        choices=PARD2_PROFILES, default="qwen3_8b")
    result.add_argument("--prompt", default="Explain speculative decoding in one paragraph.")
    result.add_argument("--max-new-tokens", type=int, default=256)
    result.add_argument("--max-cache-len", type=int, default=8192)
    result.add_argument("--page-size", type=int, default=128)
    result.add_argument("--compile-mode", default="eager",
                        choices=("eager", "default", "reduce-overhead", "max-autotune"),
                        help="Drafter compilation mode; target HIP graphs are independent.")
    result.add_argument("--td-basis-fold", action=argparse.BooleanOptionalAction,
                        default=False, help="Fold basis restoration into the TD projection.")
    result.add_argument("--fused-norm-quant", action=argparse.BooleanOptionalAction,
                        default=True, help="Fuse row RMSNorm and INT4 packing.")
    result.add_argument("--json-output", type=Path, help="Optional generation output file.")
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.mode != "ar" and not args.draft:
        argument_parser.error("--draft is required for PARD-2 generation")
    if args.mode == "pard2-td" and args.benchmark_profile == "qwen3_32b":
        argument_parser.error("Qwen3-32B supports AR and PARD2-TI in this release")
    if args.max_new_tokens <= 0 or args.max_cache_len <= 0 or args.page_size <= 0:
        argument_parser.error("token counts, cache length and page size must be positive")
    require_gptq_checkpoint(args.target)
    if args.mode != "ar":
        verify_local_resources(args.draft, args.benchmark_profile, args.mode)

    # The final runtime configuration, applied before model/cache construction.
    # Explicit caller overrides still permit inspection of the retained fallbacks.
    for name in ("QUAROT_BATCHED_H128", "QUAROT_STATIC_KV_METADATA",
                 "QUAROT_VERIFICATION_GRAPH", "QUAROT_CHUNK_PREPROCESS"):
        os.environ.setdefault(name, "1")
    runtime = load_runtime(
        mode=args.mode, target_checkpoint=args.target,
        draft_snapshot=args.draft, tokenizer_path=args.tokenizer,
        max_cache_len=args.max_cache_len, page_size=args.page_size,
        compile_mode=args.compile_mode, ignore_eos=False,
        td_basis_fold=args.td_basis_fold, fused_norm_quant=args.fused_norm_quant,
        benchmark_profile=args.benchmark_profile)
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": args.prompt}]
    if getattr(runtime.tokenizer, "chat_template", None):
        ids = runtime.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False)
    else:
        ids = runtime.tokenizer(args.prompt, return_tensors="pt").input_ids
    result = runtime.generate(ids.to("cuda"), args.max_new_tokens)
    payload = result.metrics()
    payload["mode"] = args.mode
    payload["text"] = runtime.tokenizer.decode(result.output_ids, skip_special_tokens=False)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    print(encoded)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
