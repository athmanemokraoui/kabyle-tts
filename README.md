# Kabyle TTS - ONNX Inference with Correct Tokenization

A production-ready Kabyle (Taqbaylit) text-to-speech solution using the `facebook/mms-tts-kab` ONNX model with correct MMS tokenization.

## Overview

This script works around critical bugs in the `phoonnx` library (all versions including 1.3.4a1) that prevent proper Kabyle speech synthesis:

- **Bug 1:** `VoiceConfig.from_dict()` overwrites `config["blank"] = "_"` instead of using `"|"` from the config
- **Bug 2:** `TTSTokenizer.from_phoonnx_config()` ignores config fields (`add_blank_char`, `blank_between`, etc.)
- **Bug 3:** The phoneme map doesn't include `"_"`, so `blank_id` becomes `None`
- **Bug 4:** Without valid `blank_id`, `intersperse_blank_char()` does nothing
- **Bug 5:** Result: tokens without blanks → gibberish audio

## Solution

This implementation:
1. Uses the **original MMS tokenizer** (`transformers.VitsTokenizer`) for correct tokenization
2. Runs **ONNX inference directly** via `onnxruntime`
3. Manually implements blank token insertion matching MMS VITS behavior
4. Maps missing characters (p, v, o) to closest Kabyle equivalents

## Requirements

```bash
pip install transformers torch onnxruntime-gpu huggingface_hub scipy
```

For CPU-only:
```bash
pip install transformers torch onnxruntime huggingface_hub scipy
```

## Quick Start

### In Python

```python
from kabyle_tts_suppress_mapped import KabyleTTS

# Initialize
tts = KabyleTTS(use_cuda=True)

# Synthesize
tts.speak("Azul, amek i telliḍ ?", "output.wav")

# With speed control (0.8 = slower, 1.2 = faster)
tts.speak("Taqbaylit d tutlayt tamezwarut n Lezzayer.", speed=0.8)
```

### In Google Colab

Copy the entire script into a single cell and run. All dependencies will be installed automatically.

## Character Mapping

The MMS model was trained on Kabyle text without certain characters. The following mappings are applied:

| Character | Maps To | Reason |
|---|---|---|
| `p` | `b` | 'p' not in Kabyle alphabet |
| `v` | `f` | 'v' not in Kabyle alphabet |
| `o` | `u` | Kabyle uses 'u', not 'o' |

You can customize mappings:

```python
tts = KabyleTTS(
    use_cuda=True,
    char_map={'p': 'b', 'v': 'f'}
)
```

To disable all mappings:

```python
tts = KabyleTTS(use_cuda=True, char_map={})
```

## API Reference

### `KabyleTTS`

#### `__init__(cache_dir="/content/kabyle_tts", use_cuda=True, char_map=None)`

Initialize the TTS engine.

- `cache_dir`: Directory to cache downloaded model files
- `use_cuda`: Whether to use GPU (CUDA) for inference
- `char_map`: Custom character mappings (merges with defaults)

#### `speak(text, output_path=None, speed=1.0) -> str`

Synthesize text to WAV file.

- `text`: Kabyle text to synthesize
- `output_path`: Output WAV file path (auto-generated if None)
- `speed`: Speech speed multiplier (1.0 = normal, 0.8 = slower, 1.2 = faster)
- Returns: Path to generated WAV file

#### `verify(text, suppress_mapped=True) -> bool`

Verify that tokenization matches the original MMS tokenizer.

- `text`: Text to verify
- `suppress_mapped`: If True, don't warn for texts with mapped characters
- Returns: True if tokenization matches

## Model Details

- **Model:** `facebook/mms-tts-kab` (via `OpenVoiceOS/phoonnx-mms` ONNX mirror)
- **ONNX File:** `facebook__mms-tts-kab-Amazigh/model.onnx`
- **Sample Rate:** 16,000 Hz
- **Vocab Size:** 38 symbols
- **Parameters:** 114M
- **Format:** VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)

## Example Sentences

```python
sentences = [
    "Nekk s leɛqel i tetteɣ.",
    "Ur εlimeɣ anda-tt yemma.",
    "D acu nniḍen i teččiḍ?",
    "Ur iyi-yeɛǧib ara win.",
    "Jappun teččur d tinutam!",
    "Ur lliɣ ara lsiɣ lfista.",
    "D ayen i ɣ-yecqan.",
    "Cukkeɣ uklaleɣ kra n usegzi.",
]
```

## Known Limitations

1. **16 kHz sample rate:** Lower fidelity than modern 22-24 kHz TTS models
2. **Loanword pronunciation:** Characters not in the original training data (p, v, o) are mapped to approximations
3. **Monotone delivery:** MMS models can sound robotic compared to single-language models
4. **No speaker control:** Single-speaker model

## Troubleshooting

### "CUDAExecutionProvider not available"

Install CUDA-compatible ONNX Runtime:
```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.18.0
```

### "Your browser does not support the audio element"

This is a Colab display quirk. The WAV files are valid. Download them from the Files panel or use:
```python
from IPython.display import Audio
display(Audio("output.wav"))
```

### Unintelligible audio

If you get gibberish output, the tokenization is broken. Check:
1. That you're using this script (not raw phoonnx)
2. That the model files downloaded correctly
3. That the tokenizer matches MMS (run `verify()`)

## Credits

- **Model:** Meta AI (facebook/mms-tts-kab)
- **ONNX Conversion:** OpenVoiceOS (OpenVoiceOS/phoonnx-mms)
- **Bug Analysis:** boffire / Athmane MOKRAOUI

## License

The underlying MMS model is licensed under CC-BY-NC 4.0 (non-commercial).

## Reporting Bugs

For phoonnx library bugs, report to: https://github.com/TigreGotico/phoonnx/issues

Include:
- Model: `facebook__mms-tts-kab-Amazigh`
- Bug: Missing blank tokens in tokenizer output
- Expected: `[0, 1, 0, 25, 0, 11, 0, 7, 0]` for "Azul"
- Actual: `[25, 11, 7]` (no blanks)
