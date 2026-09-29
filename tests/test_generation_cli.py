"""Portable CLI resources and argument forwarding without model allocations."""
import json
import os
from types import SimpleNamespace

import pytest

from e2e import pard2


def _arguments(mode="ar"):
    return ["--mode", mode, "--target", "checkpoints/target",
            "--tokenizer", "snapshots/tokenizer"]


@pytest.mark.parametrize("mode", ["pard2-ti", "pard2-td"])
def test_speculative_cli_requires_explicit_draft(mode, monkeypatch):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("invalid arguments must fail before loading resources")
    monkeypatch.setattr(pard2, "require_gptq_checkpoint", unexpected_load)
    with pytest.raises(SystemExit) as error:
        pard2.main(_arguments(mode))
    assert error.value.code == 2


def test_cli_rejects_unsupported_32b_td_before_loading(monkeypatch):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("unsupported TD must fail before loading resources")
    monkeypatch.setattr(pard2, "require_gptq_checkpoint", unexpected_load)
    with pytest.raises(SystemExit) as error:
        pard2.main(_arguments("pard2-td") + ["--draft", "snapshots/draft",
                                           "--profile", "qwen3_32b"])
    assert error.value.code == 2


@pytest.mark.parametrize("flag", ["--max-new-tokens", "--max-cache-len", "--page-size"])
def test_cli_rejects_nonpositive_sizes_before_loading(flag, monkeypatch):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("invalid sizes must fail before loading resources")
    monkeypatch.setattr(pard2, "require_gptq_checkpoint", unexpected_load)
    with pytest.raises(SystemExit) as error:
        pard2.main(_arguments() + [flag, "0"])
    assert error.value.code == 2


def test_draft_warp_is_required_only_for_td(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    # Sparse placeholders exercise completeness checks without allocating weights.
    with (tmp_path / "model.safetensors").open("wb") as handle:
        handle.truncate(1_000_000_000)
    pard2.verify_local_resources(tmp_path, "qwen3_8b", "pard2-ti")
    with pytest.raises(FileNotFoundError, match="warp_model.bin"):
        pard2.verify_local_resources(tmp_path, "qwen3_8b", "pard2-td")
    with (tmp_path / "warp_model.bin").open("wb") as handle:
        handle.truncate(1_000_000)
    pard2.verify_local_resources(tmp_path, "qwen3_8b", "pard2-td")


@pytest.mark.parametrize("mode", ["ar", "pard2-ti", "pard2-td"])
@pytest.mark.parametrize("chat_template", [False, True])
def test_cli_forwards_final_configuration_and_writes_generation(
        mode, chat_template, monkeypatch, tmp_path, capsys):
    events = []
    forwarded = {}
    for name in ("QUAROT_BATCHED_H128", "QUAROT_STATIC_KV_METADATA",
                 "QUAROT_VERIFICATION_GRAPH", "QUAROT_CHUNK_PREPROCESS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUAROT_STATIC_KV_METADATA", "0")

    class Tokens:
        def to(self, device):
            assert device == "cuda"
            return self

    tokens = Tokens()

    class Tokenizer:
        def __call__(self, prompt, **kwargs):
            assert prompt == "Explain INT4"
            assert kwargs == {"return_tensors": "pt"}
            events.append("plain-tokenizer")
            return SimpleNamespace(input_ids=tokens)

        def apply_chat_template(self, messages, **kwargs):
            assert messages[-1] == {"role": "user", "content": "Explain INT4"}
            assert kwargs == {"add_generation_prompt": True, "return_tensors": "pt",
                              "enable_thinking": False}
            events.append("chat-tokenizer")
            return tokens

        def decode(self, ids, **kwargs):
            assert ids == [7, 8]
            assert kwargs == {"skip_special_tokens": False}
            return "Generated text"

    tokenizer = Tokenizer()
    tokenizer.chat_template = chat_template

    def generate(ids, count):
        assert ids is tokens
        assert count == 2
        return SimpleNamespace(output_ids=[7, 8], metrics=lambda: {"generated_tokens": 2})

    def load_runtime(**kwargs):
        forwarded.update(kwargs)
        events.append("load")
        return SimpleNamespace(tokenizer=tokenizer, generate=generate)

    monkeypatch.setattr(pard2, "require_gptq_checkpoint",
                        lambda path: events.append(("gptq", path)))
    monkeypatch.setattr(pard2, "verify_local_resources",
                        lambda *args: events.append(("draft", args)))
    monkeypatch.setattr(pard2, "load_runtime", load_runtime)
    output = tmp_path / "nested" / "generation.json"
    arguments = _arguments(mode) + ["--prompt", "Explain INT4", "--max-new-tokens", "2",
                                   "--json-output", str(output)]
    if mode != "ar":
        arguments += ["--draft", "snapshots/draft"]
    pard2.main(arguments)

    assert events[0] == ("gptq", "checkpoints/target")
    if mode == "ar":
        assert events[1] == "load"
    else:
        assert events[1] == ("draft", ("snapshots/draft", "qwen3_8b", mode))
    assert forwarded == {
        "mode": mode, "target_checkpoint": "checkpoints/target",
        "draft_snapshot": None if mode == "ar" else "snapshots/draft",
        "tokenizer_path": "snapshots/tokenizer", "max_cache_len": 8192,
        "page_size": 128, "compile_mode": "eager", "ignore_eos": False,
        "td_basis_fold": False, "fused_norm_quant": True,
        "benchmark_profile": "qwen3_8b",
    }
    assert events[-1] == ("chat-tokenizer" if chat_template else "plain-tokenizer")
    assert os.environ["QUAROT_STATIC_KV_METADATA"] == "0"
    for name in ("QUAROT_BATCHED_H128", "QUAROT_VERIFICATION_GRAPH", "QUAROT_CHUNK_PREPROCESS"):
        assert os.environ[name] == "1"
    expected = {"generated_tokens": 2, "mode": mode, "text": "Generated text"}
    assert json.loads(output.read_text()) == expected
    assert json.loads(capsys.readouterr().out) == expected
