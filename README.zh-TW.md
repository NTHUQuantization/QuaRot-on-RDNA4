# QuaRot on RDNA 4

**針對量化大型語言模型推論的架構感知加速**

[English](README.md) · [架構說明（英文）](docs/architecture.md) · [Checkpoint 準備（英文）](docs/checkpoints.md)

本 repository 收錄國立清華大學專題報告 *QuaRot on RDNA 4: Architecture-Aware Acceleration for Quantized LLM Inference* 的實作。系統將 QuaRot 移植至 AMD ROCm/HIP，支援 dense Qwen3，結合 GPTQ W4A4KV4 推論、RDNA 4 INT4 WMMA 核心、算子融合，以及 PARD-2 speculative decoding。

主要平台為 **AMD Radeon AI PRO R9700（32 GB、`gfx1201`）**。此版本保留最終實作、checkpoint 轉換、編譯原始碼及正確性檢查；模型權重、benchmark 資料、量測輸出、實驗機專用腳本及未採用的實驗不納入 repository。

## 支援範圍

| Target | Model profile | 自回歸 AR | PARD-2 TI | PARD-2 TD |
| --- | --- | --- | --- | --- |
| Qwen3-8B | `qwen3_8b` | 支援 | 支援 | 支援 |
| Qwen3-14B | `qwen3_14b` | 支援 | 支援 | 支援 |
| Qwen3-32B | `qwen3_32b` | 支援 | 支援 | — |

三種 target 皆採 GPTQ 權重量化、4-bit activations，以及保留原生 GQA head 數量的 paged 4-bit KV cache。正規化、非線性運算與部分其他運算仍使用浮點數；drafter 採 BF16。

正式生成流程為單一請求、greedy decoding、關閉 thinking、正常 EOS 終止。PARD-2 每輪平行產生 15 個候選 token，首次驗證處理 15 個位置，後續連同 pending token 處理 16 個位置。TI 共用 `amd/PARD2-Qwen3-8B` drafter；TD 使用對應 target 大小的 drafter。

## 編譯

請先準備具有 ROCm PyTorch、HIP compiler，以及相容 ROCm FlashAttention 的 Linux 環境。專題使用的環境為 **ROCm 7.2、PyTorch 2.9.1、Transformers 4.57.6**。PyTorch 在 ROCm 上仍沿用 `torch.cuda` API，因此指令與原始碼中的 `cuda` 在此環境指向 AMD GPU。

```bash
git clone https://github.com/NTHUQuantization/QuaRot-on-RDNA4.git
cd QuaRot-on-RDNA4

python -m pip install -r requirements.txt
python -m pip install -e third-party/hadacore --no-build-isolation
python -m pip install -e . --no-build-isolation
```

預設編譯目標為 `gfx1201`。修改 HIP 原始碼或更換 ROCm/PyTorch 工具鏈後，請重新編譯相應 extension。`quarot` 使用本 repository 提供的 `fast_hadamard_transform` extension，必須先完成其編譯。

## 準備模型

依照 [checkpoint 指南](docs/checkpoints.md) 下載固定版本的 dense target 與 drafter，再將 target 轉換成 GPTQ checkpoint。此 runtime 需要一致的 grouped Hadamard 格式、rotation metadata、Qwen3 Q/K norm 與量化設定，不能直接替換成任意 INT4 checkpoint。

以下以 Qwen3-8B 為例，請將路徑替換成本機目錄：

```bash
SOURCE=/path/to/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
TARGET=/path/to/qwen3-8b-quarot-gptq
DRAFT=/path/to/models--amd--PARD2-Qwen3-8B/snapshots/67a1516c8f6fc145cda99916799a0cbb3a4af135
```

推論只載入指定的本機資源，不會自動下載模型或選擇實驗機路徑。

## 生成

從 repository 根目錄執行。CLI 預設啟用以下最終 runtime 設定；明確設定可覆蓋 shell 中既有的設定：

```bash
export QUAROT_BATCHED_H128=1
export QUAROT_STATIC_KV_METADATA=1
export QUAROT_VERIFICATION_GRAPH=1
export QUAROT_CHUNK_PREPROCESS=1
```

使用 fused target 進行 AR 生成：

```bash
python -m e2e.pard2 \
  --mode ar --profile qwen3_8b \
  --target "$TARGET" --tokenizer "$SOURCE" \
  --compile-mode eager --fused-norm-quant \
  --max-cache-len 4096 --max-new-tokens 256 \
  --prompt 'Explain speculative decoding in one paragraph.'
```

使用 PARD-2 TI：

```bash
python -m e2e.pard2 \
  --mode pard2-ti --profile qwen3_8b \
  --target "$TARGET" --tokenizer "$SOURCE" --draft "$DRAFT" \
  --compile-mode eager --fused-norm-quant \
  --max-cache-len 4096 --max-new-tokens 256 \
  --prompt 'Explain speculative decoding in one paragraph.'
```

8B TD 將模式改為 `--mode pard2-td`。14B 改用 `qwen3_14b` 及相應 target/tokenizer，TD 使用 14B drafter，TI 仍使用 8B drafter。32B 改用 `qwen3_32b`，支援 AR 與 TI。

`--profile` 選擇模型設定，並保留 `--benchmark-profile` 作為相容別名。`--compile-mode eager` 控制 drafter 編譯，與 target HIP graph 開關分開。程式輸出包含生成文字及執行資訊的 JSON；cache 容量需容納 prompt、生成 token 及 speculative verification 位置。

## 原始碼與文件

| 路徑 | 用途 |
| --- | --- |
| `quarot/kernels/` | INT4 WMMA GEMM、算子融合、量化及 paged attention |
| `quarot/nn/`、`quarot/transformers/` | INT4 layer、online rotation、KV4 cache 與 runtime 整合 |
| `e2e/quantized_qwen3/` | Qwen3 量化模型 |
| `e2e/checkpoint_utils/` | Streaming GPTQ 與 rotation metadata |
| `e2e/speculative.py` | PARD-2 drafting、verification、acceptance 與 TD feature |
| `e2e/verification_graph.py` | 固定驗證形狀的 graph capture/replay |
| `third-party/hadacore/` | 最終整合 runtime 的 HIP Hadamard extension |
| `components/wave/` | 獨立保留的最終 Wave Hadamard component |
| `tests/` | 正確性與 runtime contract 檢查 |

獨立 Wave component 與整合推論 runtime 保留各自的 dispatch；預設 inference build 使用 `third-party/hadacore/`。詳細的算子對應、資料流與兩者差異請見 [架構說明](docs/architecture.md)。

## 正確性檢查

完成 extension 編譯後，安裝測試依賴並執行：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

GPU 檢查需要上述 ROCm 環境。獨立 Wave component 的編譯及測試方式見其 [README](components/wave/README.md)。

## 授權與致謝

本專題以 [QuaRot](https://github.com/spcl/QuaRot)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer)、[HadaCore](https://arxiv.org/abs/2412.08832)、[PARD/PARD-2](https://github.com/AMD-AGI/PARD)，以及 Transformers 的 Qwen3 實作為基礎。Repository 保留 QuaRot 的 [Apache-2.0 授權](LICENSE)；各元件適用的授權（包含衍生自 QuIP-sharp 的 Hadamard 工具）見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型與資料集各自適用其原有授權。
