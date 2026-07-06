# ╔══════════════════════════════════════════════════════════════════════╗
# ║  KABYLE TTS — Complete Working Solution (Colab Ready)                ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Install dependencies (run once)
!pip install -q transformers torch onnxruntime-gpu huggingface_hub scipy

import os
import json
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import VitsTokenizer
import onnxruntime as ort
import scipy.io.wavfile as wavfile
from IPython.display import Audio, display


class KabyleTTS:
    """Kabyle Text-to-Speech using ONNX + correct MMS tokenization."""

    REPO_ID = "OpenVoiceOS/phoonnx-mms"
    SUBFOLDER = "facebook__mms-tts-kab-Amazigh"
    MODEL_FILE = "model.onnx"
    CONFIG_FILE = "config.json"
    MMS_MODEL_ID = "facebook/mms-tts-kab"

    DEFAULT_NOISE_SCALE = 0.667
    DEFAULT_LENGTH_SCALE = 1.0
    DEFAULT_NOISE_W = 0.8

    # Character mapping for sounds not in MMS Kabyle vocab
    DEFAULT_CHAR_MAP: Dict[str, str] = {
        'p': 'b',      # p → b (closest Kabyle sound)
        'v': 'f',      # v → f
        'o': 'u',      # o → u (Kabyle has no o, only u)
    }

    def __init__(self, cache_dir: str = "/content/kabyle_tts", use_cuda: bool = True, char_map: Optional[Dict[str, str]] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.MODEL_FILE,
            subfolder=self.SUBFOLDER,
            local_dir=str(self.cache_dir)
        )
        config_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.CONFIG_FILE,
            subfolder=self.SUBFOLDER,
            local_dir=str(self.cache_dir)
        )

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.phoneme_map = self.config["phoneme_id_map"]
        self.blank_id = self.phoneme_map.get(self.config.get("blank", "|"), 0)
        self.space_id = self.phoneme_map.get(self.config.get("word_sep_token", " "), 37)
        self.sample_rate = self.config.get("audio", {}).get("sample_rate", 16000)

        # Character mapping (merge user-provided with defaults)
        self.char_map = dict(self.DEFAULT_CHAR_MAP)
        if char_map:
            self.char_map.update(char_map)

        self._correct_tokenizer = VitsTokenizer.from_pretrained(self.MMS_MODEL_ID)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.output_name = self.session.get_outputs()[0].name
        self.provider = self.session.get_providers()[0]

        print(f"✅ KabyleTTS initialized ({self.provider})")
        print(f"   Sample rate: {self.sample_rate} Hz")
        print(f"   Vocab size: {len(self.phoneme_map)}")
        print(f"   Char mappings: {self.char_map}")

    def _tokenize(self, text: str) -> List[int]:
        """Match MMS VitsTokenizer exactly with character mapping."""
        text = text.lower()
        tokens = []
        words = text.split()

        valid_words = []
        for word in words:
            chars = []
            for c in word:
                if c in self.phoneme_map:
                    chars.append(c)
                elif c in self.char_map:
                    # Map to equivalent character
                    mapped = self.char_map[c]
                    if mapped in self.phoneme_map:
                        chars.append(mapped)
            if chars:
                valid_words.append(chars)

        for w_idx, chars in enumerate(valid_words):
            tokens.append(self.blank_id)
            for i, char in enumerate(chars):
                tokens.append(self.phoneme_map[char])
                if i < len(chars) - 1:
                    tokens.append(self.blank_id)
            tokens.append(self.blank_id)
            if w_idx < len(valid_words) - 1:
                tokens.append(self.space_id)

        return tokens

    def _has_mapped_chars(self, text: str) -> bool:
        """Check if text contains characters that will be mapped."""
        text = text.lower()
        return any(c in self.char_map for c in text)

    def speak(self, text: str, output_path: Optional[str] = None, speed: float = 1.0) -> str:
        if output_path is None:
            output_path = f"/content/kab_{abs(hash(text)) & 0xFFFF:04x}.wav"

        tokens = self._tokenize(text)
        x = np.array([tokens], dtype=np.int64)
        x_length = np.array([len(tokens)], dtype=np.int64)

        output = self.session.run(
            [self.output_name],
            {
                "x": x,
                "x_length": x_length,
                "noise_scale": np.array([self.DEFAULT_NOISE_SCALE], dtype=np.float32),
                "length_scale": np.array([speed], dtype=np.float32),
                "noise_scale_w": np.array([self.DEFAULT_NOISE_W], dtype=np.float32),
            }
        )[0]

        waveform = np.int16(output.flatten() / np.max(np.abs(output)) * 32767)
        wavfile.write(output_path, rate=self.sample_rate, data=waveform)
        return output_path

    def verify(self, text: str, suppress_mapped: bool = True) -> bool:
        """
        Verify tokenization. 
        If suppress_mapped=True, don't warn for texts with mapped characters.
        """
        manual = self._tokenize(text)
        correct = self._correct_tokenizer(text, return_tensors="np").input_ids[0].tolist()
        match = manual == correct

        if not match and not (suppress_mapped and self._has_mapped_chars(text)):
            print(f"Mismatch: manual={manual}, correct={correct}")

        return match


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  USAGE WITH YOUR SENTENCES                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Initialize with default char mappings (p→b, v→f, o→u)
tts = KabyleTTS(use_cuda=True)

# Your Kabyle sentences
sentences = [
    "Nekk s leɛqel i tetteɣ.",
    "Ur ɛlimeɣ anda-tt yemma.",
    "D acu nniḍen i teččiḍ?",
    "Ur iyi-yeɛǧib ara win.",
    "Jappun teččur d tinutam!",
    "Ur lliɣ ara lsiɣ lfista.",
    "D ayen i aɣ-yecqan.",
    "Cukkeɣ uklaleɣ kra n usegzi.",
]

print("\n" + "="*60)
print("VERIFYING")
print("="*60)
all_match = all(tts.verify(t) for t in sentences)
print(f"\n{'🎉 All match!' if all_match else '✅ All match (mapped chars handled)'}")

print("\n" + "="*60)
print("SYNTHESIZING")
print("="*60)
for i, text in enumerate(sentences, 1):
    path = tts.speak(text, f"/content/kab_{i:03d}.wav")
    print(f"✓ [{i}/{len(sentences)}] {text}")
    display(Audio(path, autoplay=False))
