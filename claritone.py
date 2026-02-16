"""
Ultimate Audio Enhancement Pipeline - GPU Accelerated Version
==============================================================
CUDA/GPU accelerated for maximum quality AND speed.

FIXED: Stereo preservation - processes L/R channels independently instead of M/S

Requires:
  pip install cupy-cuda12x torch torchaudio

Falls back to CPU if CUDA is not available.
"""

import argparse
import inspect
import glob
import os
import sys
import re

# ── Organized directory layout ────────────────────────────────────────
# Everything lives next to the script:
#
#   <script_dir>/
#   ├── wizard5_gpu_blackbox.py
#   ├── models/
#   │   ├── apollo/          ← Apollo checkpoint + config (user-supplied)
#   │   ├── demucs/          ← Demucs / torch-hub cache
#   │   └── msst/            ← Music-Source-Separation-Training repo (auto-cloned)
#   ├── cache/               ← Temp processing artefacts
#   └── output/              ← Final enhanced files (default --output-dir)
#
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List
import numpy as np
import soundfile as sf
import math
from scipy import signal, ndimage
import warnings
warnings.filterwarnings('ignore')

def _patch_scipy_iirfilter_safety():
    """Clamp normalized digital critical frequencies into (0,1) to avoid SciPy ValueError.

    Some libraries (notably AudioSR's lowpass helper) can occasionally compute Wn as exactly 0 or 1
    for 44.1k material, which SciPy rejects. Clamping by a tiny epsilon is sonically negligible
    but prevents hard crashes.
    """
    try:
        from scipy.signal import _filter_design as _fd  # type: ignore
    except Exception:
        return

    if getattr(_fd, "__W5_IIRFILTER_PATCHED__", False):
        return

    _orig_iirfilter = _fd.iirfilter

    def _iirfilter_safe(N, Wn, *args, **kwargs):
        try:
            analog = bool(kwargs.get("analog", False))
            fs = kwargs.get("fs", None)
            if (not analog) and (fs is None) and (Wn is not None):
                import numpy as _np
                W = _np.asarray(Wn, dtype=float)
                eps = 1e-6
                W = _np.clip(W, eps, 1.0 - eps)
                if _np.ndim(Wn) == 0:
                    Wn = float(W)
                else:
                    Wn = W.tolist()
        except Exception:
            # Fall through to original call
            pass
        return _orig_iirfilter(N, Wn, *args, **kwargs)

    _fd.iirfilter = _iirfilter_safe
    _fd.__W5_IIRFILTER_PATCHED__ = True


_patch_scipy_iirfilter_safety()

import subprocess
import tempfile
import shutil
import time
import struct

# =========================
# GPU Backend Detection
# =========================

GPU_AVAILABLE = False
BACKEND = "cpu"

# Try CuPy first (fastest for signal processing)
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    from cupyx.scipy import ndimage as cp_ndimage
    from cupyx.scipy import fft as cp_fft
    GPU_AVAILABLE = True
    BACKEND = "cupy"
    print("✓ CuPy CUDA backend available")
except ImportError:
    cp = None
    print("○ CuPy not available")

# Try PyTorch as fallback
try:
    import torch
    import torchaudio
    if torch.cuda.is_available():
        if not GPU_AVAILABLE:
            GPU_AVAILABLE = True
            BACKEND = "torch"
        print(f"✓ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("○ PyTorch available but no CUDA")
except ImportError:
    torch = None
    print("○ PyTorch not available")

# Import CPU libraries as fallback
import scipy.fft
from scipy.fft import fft, ifft, rfft, irfft

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("○ librosa not available (some features disabled)")

from numba import jit, prange

print(f"\n>>> Using backend: {BACKEND.upper()}")
if GPU_AVAILABLE:
    print(">>> GPU acceleration: ENABLED")
else:
    print(">>> GPU acceleration: DISABLED (CPU fallback)")
print()


# =========================
# GPU Array Utilities
# =========================

def to_gpu(arr: np.ndarray):
    """Move array to GPU"""
    # Ensure contiguous array (fixes negative stride issues)
    arr = np.ascontiguousarray(arr)
    if BACKEND == "cupy" and cp is not None:
        return cp.asarray(arr)
    elif BACKEND == "torch" and torch is not None:
        return torch.from_numpy(arr).cuda()
    return arr

def to_cpu(arr) -> np.ndarray:
    """Move array to CPU"""
    if BACKEND == "cupy" and cp is not None:
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    elif BACKEND == "torch" and torch is not None:
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
    return np.asarray(arr)


# =========================
# Flexible I/O: any format in, any format out
# =========================

STANDARD_SAMPLE_RATES = [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000,
                         88200, 96000, 176400, 192000]


def _detect_bit_depth(path: str) -> int:
    """Detect the bit depth of an audio file.

    Returns 16, 24, or 32 (float).  Falls back to 16 if unknown.
    """
    try:
        info = sf.info(path)
        subtype = info.subtype.upper()
        if 'FLOAT' in subtype or 'F32' in subtype:
            return 32
        elif '24' in subtype:
            return 24
        elif '8' in subtype:
            return 16          # upgrade 8-bit to our minimum
        elif '16' in subtype:
            return 16
        else:
            return 16
    except Exception:
        pass
    # MP3 / AAC / OGG → treat as 16
    return 16


def read_audio_any_format(input_path: str) -> tuple:
    """Read audio from any standard format (WAV, FLAC, MP3, OGG, AIFF, AAC …).

    Returns ``(audio_float64_2d, sample_rate, detected_bit_depth)``.
    Audio is always float64 in [-1, 1], shape ``(N, C)``.
    """
    ext = os.path.splitext(input_path)[1].lower()

    # --- Try soundfile first (handles WAV, FLAC, AIFF, OGG natively) ---
    try:
        audio, sr = sf.read(input_path, always_2d=True)
        bit_depth = _detect_bit_depth(input_path)
        return audio.astype(np.float64), int(sr), bit_depth
    except Exception:
        pass

    # --- Fallback: torchaudio (handles MP3, AAC, OGG, OPUS via ffmpeg) ---
    if torch is not None:
        try:
            import torchaudio as _ta
            waveform, sr_ta = _ta.load(input_path)
            # torchaudio returns (channels, samples) as float32 in [-1, 1]
            audio = waveform.numpy().T.astype(np.float64)   # → (samples, channels)
            if audio.ndim == 1:
                audio = audio[:, np.newaxis]
            bit_depth = 16 if ext in ('.mp3', '.aac', '.m4a', '.ogg', '.opus', '.wma') else 24
            return audio, int(sr_ta), bit_depth
        except Exception:
            pass

    # --- Last resort: ffmpeg subprocess → temp WAV → soundfile ---
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_wav = os.path.join(td, "converted.wav")
            subprocess.check_call(
                ["ffmpeg", "-y", "-i", input_path, "-acodec", "pcm_f32le", tmp_wav],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            audio, sr = sf.read(tmp_wav, always_2d=True)
            bit_depth = 16 if ext in ('.mp3', '.aac', '.m4a', '.ogg', '.opus', '.wma') else 24
            return audio.astype(np.float64), int(sr), bit_depth
    except Exception:
        pass

    raise RuntimeError(f"Cannot read audio file: {input_path}")


def write_audio_any_format(output_path: str, audio: np.ndarray, sr: int,
                           bit_depth: int = 24, output_format: str = 'wav') -> str:
    """Write audio in the specified format and bit depth.

    Supports: wav, flac, mp3.  Returns actual output path (extension may change).
    """
    fmt = output_format.lower().strip('.')
    base, _ = os.path.splitext(output_path)
    output_path = f"{base}.{fmt}"

    if fmt in ('wav', 'flac'):
        subtype_map = {
            ('wav', 16): 'PCM_16', ('wav', 24): 'PCM_24', ('wav', 32): 'FLOAT',
            ('flac', 16): 'PCM_16', ('flac', 24): 'PCM_24', ('flac', 32): 'PCM_24',
        }
        subtype = subtype_map.get((fmt, bit_depth), 'PCM_24')
        sf.write(output_path, audio, sr, subtype=subtype)
    elif fmt == 'mp3':
        with tempfile.TemporaryDirectory() as td:
            tmp_wav = os.path.join(td, "tmp.wav")
            sf.write(tmp_wav, audio, sr, subtype='PCM_24')
            bitrate = '320k' if bit_depth >= 24 else '192k'
            try:
                subprocess.check_call(
                    ["ffmpeg", "-y", "-i", tmp_wav, "-b:a", bitrate, "-q:a", "0",
                     output_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                print("  ⚠️  ffmpeg not found; falling back to WAV output")
                output_path = f"{base}.wav"
                sf.write(output_path, audio, sr, subtype='PCM_24')
    else:
        output_path = f"{base}.wav"
        sf.write(output_path, audio, sr, subtype='FLOAT')

    return output_path


def _ensure_finite_audio(x: np.ndarray, clamp: float = 1.0) -> np.ndarray:
    """Ensure audio has no NaN/Inf and is within [-clamp, clamp].

    If x is a path/string (some AudioSR versions accept file paths), this is a no-op.
    """
    if x is None:
        return x

    # If AudioSR is fed a filename/path, don't touch it.
    if isinstance(x, (str, bytes, os.PathLike)):
        return x

    x = np.asarray(x)

    # If array is not numeric (e.g., string/object), don't touch it.
    if x.dtype.kind not in ("f", "i", "u", "c"):
        return x

    # Replace NaN/Inf with 0 to satisfy downstream validators (librosa)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Hard clamp to prevent numeric explosions
    if clamp is not None:
        x = np.clip(x, -float(clamp), float(clamp))

    return x

def _normalize_chunk_shape_for_ola(out_chunk: np.ndarray, target_len: int, target_ch: int) -> np.ndarray:
    """Normalize AudioSR chunk output for overlap-add.

    Ensures shape (L, C) where L<=target_len and C==target_ch.
    Handles common returns:
      - (L,) mono
      - (C, L) channel-first
      - (L, C) already correct
    Also forces float32 to keep memory down.
    """
    if isinstance(out_chunk, (str, bytes, os.PathLike)):
        raise TypeError("AudioSR returned a path/string for chunk; expected numeric audio buffer.")
    x = np.asarray(out_chunk)
    # If object/string array slipped through, bail early
    if x.dtype.kind not in ("f", "i", "u", "c"):
        raise TypeError(f"AudioSR chunk has non-numeric dtype: {x.dtype}")
    # Squeeze singleton dims
    x = np.squeeze(x)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim == 2:
        # If channel-first (C, L) and C is small, transpose
        if x.shape[0] in (1, 2) and x.shape[1] > x.shape[0] and target_ch == x.shape[0]:
            x = x.T
        # If it's (L, C) but C doesn't match, try another transpose heuristic
        elif x.shape[1] in (1, 2) and x.shape[0] > x.shape[1] and target_ch == x.shape[1]:
            pass
        elif x.shape[0] in (1, 2) and target_ch == x.shape[0] and x.shape[1] >= 8:
            x = x.T
    else:
        # Unexpected high-rank output; flatten last dim as channels if possible
        x = x.reshape(x.shape[0], -1)

    # Match channel count
    if x.shape[1] != target_ch:
        if target_ch == 1:
            x = np.mean(x, axis=1, keepdims=True)
        elif x.shape[1] == 1 and target_ch == 2:
            x = np.repeat(x, 2, axis=1)
        else:
            # Best effort: take first target_ch channels or pad
            if x.shape[1] > target_ch:
                x = x[:, :target_ch]
            else:
                pad = np.zeros((x.shape[0], target_ch - x.shape[1]), dtype=x.dtype)
                x = np.concatenate([x, pad], axis=1)

    # Limit length
    L = min(int(target_len), int(x.shape[0]))
    x = x[:L, :]

    return x.astype(np.float32, copy=False)


def _patch_librosa_for_audiosr():
    """Monkeypatch librosa.stft/istft to sanitize non-finite buffers.

    AudioSR sometimes produces NaN/Inf internally; librosa will hard-error on that.
    We sanitize the input to stft/istft so the pipeline can continue.
    """
    try:
        import librosa as _librosa
        import numpy as _np
        try:
            import torch as _torch
        except Exception:
            _torch = None
    except Exception:
        return

    if getattr(_librosa, "__W5_PATCHED__", False):
        return

    _orig_stft = _librosa.stft

    def _stft_safe(y, *args, **kwargs):
        # Convert torch -> numpy if needed
        try:
            if _torch is not None and isinstance(y, _torch.Tensor):
                y = y.detach().cpu().numpy()
        except Exception:
            pass
        # Sanitize numeric buffers
        try:
            y = _np.asarray(y)
            if y.dtype.kind in ("f", "i", "u", "c"):
                y = _np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
        return _orig_stft(y, *args, **kwargs)

    _librosa.stft = _stft_safe

    # Keep symmetry if AudioSR (or user code) calls istft
    try:
        _orig_istft = _librosa.istft

        def _istft_safe(stft_matrix, *args, **kwargs):
            try:
                M = _np.asarray(stft_matrix)
                if M.dtype.kind in ("f", "i", "u", "c"):
                    M = _np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                M = stft_matrix
            return _orig_istft(M, *args, **kwargs)

        _librosa.istft = _istft_safe
    except Exception:
        pass

    _librosa.__W5_PATCHED__ = True


# =========================
# AudioSR Lowpass Safety Patch
# =========================

_AUDIOSR_LOWPASS_PATCHED = False

def _patch_audiosr_lowpass():
    """Clamp AudioSR lowpass critical frequencies to (0,1).

    AudioSR's internal ellip() design will crash if normalized Wn is exactly
    0 or 1 (or non-finite). Some AudioSR versions can produce hi==1.0 on
    44.1k material, causing:
      ValueError: Digital filter critical frequencies must be 0 < Wn < 1

    This monkeypatch is safe: it only nudges invalid values to a valid
    interior range and otherwise leaves AudioSR behavior unchanged.
    """
    global _AUDIOSR_LOWPASS_PATCHED
    if _AUDIOSR_LOWPASS_PATCHED:
        return

    try:
        import audiosr.lowpass as _lp
    except Exception:
        return

    if getattr(_lp, '__W5_PATCHED__', False):
        _AUDIOSR_LOWPASS_PATCHED = True
        return

    try:
        orig = _lp.lowpass_filter
    except Exception:
        return

    def _clamp01(x):
        try:
            import numpy as _np
            v = float(x)
            if not _np.isfinite(v):
                return 0.5
            if v <= 0.0:
                return 1e-3
            if v >= 1.0:
                return 0.999
            # Keep a tiny margin from the endpoints
            if v < 1e-6:
                return 1e-3
            if v > 0.999999:
                return 0.999
            return v
        except Exception:
            return x

    def lowpass_filter_safe(*args, **kwargs):
        # AudioSR lowpass_filter signature is typically: (audio, sr, hi, ...)
        if len(args) >= 3:
            try:
                a = list(args)
                a[2] = _clamp01(a[2])
                args = tuple(a)
            except Exception:
                pass

        # Some variants may use named params
        for k in ('hi', 'Wn'):
            if k in kwargs:
                kwargs[k] = _clamp01(kwargs[k])

        return orig(*args, **kwargs)

    try:
        _lp.lowpass_filter = lowpass_filter_safe
        _lp.__W5_PATCHED__ = True
        _AUDIOSR_LOWPASS_PATCHED = True
    except Exception:
        return


def gpu_fft(x, n=None, axis=-1):
    """GPU-accelerated FFT"""
    if BACKEND == "cupy" and cp is not None:
        x_gpu = cp.asarray(x)
        return cp.fft.fft(x_gpu, n=n, axis=axis)
    elif BACKEND == "torch" and torch is not None:
        x_gpu = torch.from_numpy(np.ascontiguousarray(x).astype(np.complex128)).cuda()
        result = torch.fft.fft(x_gpu, n=n, dim=axis)
        return result.cpu().numpy()
    return np.fft.fft(x, n=n, axis=axis)

def gpu_ifft(x, n=None, axis=-1):
    """GPU-accelerated IFFT"""
    if BACKEND == "cupy" and cp is not None:
        x_gpu = cp.asarray(x)
        return cp.fft.ifft(x_gpu, n=n, axis=axis)
    elif BACKEND == "torch" and torch is not None:
        x_gpu = torch.from_numpy(np.ascontiguousarray(np.asarray(x))).cuda()
        result = torch.fft.ifft(x_gpu, n=n, dim=axis)
        return result.cpu().numpy()
    return np.fft.ifft(x, n=n, axis=axis)

def gpu_rfft(x, n=None, axis=-1):
    """GPU-accelerated real FFT"""
    if BACKEND == "cupy" and cp is not None:
        x_gpu = cp.asarray(x)
        return cp.fft.rfft(x_gpu, n=n, axis=axis)
    elif BACKEND == "torch" and torch is not None:
        x_gpu = torch.from_numpy(np.ascontiguousarray(x)).cuda()
        result = torch.fft.rfft(x_gpu, n=n, dim=axis)
        return result.cpu().numpy()
    return np.fft.rfft(x, n=n, axis=axis)

def gpu_irfft(x, n=None, axis=-1):
    """GPU-accelerated inverse real FFT"""
    if BACKEND == "cupy" and cp is not None:
        x_gpu = cp.asarray(x)
        return cp.asnumpy(cp.fft.irfft(x_gpu, n=n, axis=axis))
    elif BACKEND == "torch" and torch is not None:
        x_gpu = torch.from_numpy(np.ascontiguousarray(np.asarray(x))).cuda()
        result = torch.fft.irfft(x_gpu, n=n, dim=axis)
        return result.cpu().numpy()
    return np.fft.irfft(x, n=n, axis=axis)


# =========================
# GPU-Accelerated STFT
# =========================

class GPUStft:
    """GPU-accelerated Short-Time Fourier Transform"""
    
    def __init__(self, n_fft: int = 2048, hop_length: int = 512, 
                 window: str = 'hann'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_type = window
        self.window = signal.get_window(window, n_fft).astype(np.float64)
        
        if BACKEND == "cupy" and cp is not None:
            self.window_gpu = cp.asarray(self.window)
        elif BACKEND == "torch" and torch is not None:
            self.window_gpu = torch.from_numpy(np.ascontiguousarray(self.window)).cuda()
        else:
            self.window_gpu = self.window
    
    def stft(self, x: np.ndarray) -> np.ndarray:
        """Compute STFT on GPU"""
        if BACKEND == "cupy" and cp is not None:
            return self._stft_cupy(x)
        elif BACKEND == "torch" and torch is not None:
            return self._stft_torch(x)
        else:
            return self._stft_cpu(x)
    
    def istft(self, X: np.ndarray, length: Optional[int] = None) -> np.ndarray:
        """Compute inverse STFT on GPU"""
        if BACKEND == "cupy" and cp is not None:
            return self._istft_cupy(X, length)
        elif BACKEND == "torch" and torch is not None:
            return self._istft_torch(X, length)
        else:
            return self._istft_cpu(X, length)
    
    def _stft_cupy(self, x: np.ndarray) -> np.ndarray:
        x_gpu = cp.asarray(x)
        n_frames = 1 + (len(x) - self.n_fft) // self.hop_length
        
        # Create frame matrix on GPU
        indices = cp.arange(self.n_fft)[None, :] + cp.arange(n_frames)[:, None] * self.hop_length
        frames = x_gpu[indices] * self.window_gpu
        
        # FFT on GPU
        X = cp.fft.rfft(frames, axis=1).T
        return cp.asnumpy(X)
    
    def _stft_torch(self, x: np.ndarray) -> np.ndarray:
        x_gpu = torch.from_numpy(np.ascontiguousarray(x)).cuda()
        X = torch.stft(
            x_gpu, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window_gpu,
            return_complex=True
        )
        return X.cpu().numpy()
    
    def _stft_cpu(self, x: np.ndarray) -> np.ndarray:
        if LIBROSA_AVAILABLE:
            return librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                               window=self.window_type)
        else:
            n_frames = 1 + (len(x) - self.n_fft) // self.hop_length
            X = np.zeros((self.n_fft // 2 + 1, n_frames), dtype=np.complex128)
            for i in range(n_frames):
                start = i * self.hop_length
                frame = x[start:start + self.n_fft] * self.window
                X[:, i] = np.fft.rfft(frame)
            return X
    
    def _istft_cupy(self, X: np.ndarray, length: Optional[int]) -> np.ndarray:
        X_gpu = cp.asarray(X)
        n_frames = X.shape[1]
        
        if length is None:
            length = (n_frames - 1) * self.hop_length + self.n_fft
        
        # IFFT on GPU
        frames = cp.fft.irfft(X_gpu.T, n=self.n_fft, axis=1)
        frames = frames * self.window_gpu
        
        # Overlap-add on GPU
        output = cp.zeros(length, dtype=cp.float64)
        window_sum = cp.zeros(length, dtype=cp.float64)
        
        for i in range(n_frames):
            start = i * self.hop_length
            end = min(start + self.n_fft, length)
            frame_len = end - start
            output[start:end] += frames[i, :frame_len]
            window_sum[start:end] += self.window_gpu[:frame_len] ** 2
        
        # Normalize
        window_sum = cp.maximum(window_sum, 1e-8)
        output = output / window_sum
        
        return cp.asnumpy(output)
    
    def _istft_torch(self, X: np.ndarray, length: Optional[int]) -> np.ndarray:
        X_gpu = torch.from_numpy(np.ascontiguousarray(X)).cuda()
        result = torch.istft(
            X_gpu,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window_gpu,
            length=length
        )
        return result.cpu().numpy()
    
    def _istft_cpu(self, X: np.ndarray, length: Optional[int]) -> np.ndarray:
        if LIBROSA_AVAILABLE:
            return librosa.istft(X, hop_length=self.hop_length, 
                                window=self.window_type, length=length)
        else:
            n_frames = X.shape[1]
            if length is None:
                length = (n_frames - 1) * self.hop_length + self.n_fft
            
            output = np.zeros(length)
            window_sum = np.zeros(length)
            
            for i in range(n_frames):
                start = i * self.hop_length
                frame = np.fft.irfft(X[:, i], n=self.n_fft) * self.window
                end = min(start + self.n_fft, length)
                frame_len = end - start
                output[start:end] += frame[:frame_len]
                window_sum[start:end] += self.window[:frame_len] ** 2
            
            window_sum = np.maximum(window_sum, 1e-8)
            return output / window_sum


# =========================
# Configuration
# =========================

class UltimateGPUConfig:
    """Configuration optimized for GPU processing"""
    
    # DSRE - Very conservative: only subtle detail recovery, no harshness
    DSRE_M = 8                     # Fewer stages = less artifacts
    DSRE_DECAY = 1.8               # Fast decay = gentle
    DSRE_PRE_HP = 5000.0           # Only touch content above 5kHz
    DSRE_POST_HP = 15000.0         # Only keep DSRE output above 14kHz (fills gap)
    DSRE_FILTER_ORDER = 4          # Very gentle slope
    
    # =========================================================================
    # ANALOG VITALITY — replaces HFR
    # =========================================================================
    # Philosophy: instead of synthesizing HF content, use gentle saturation to
    # generate natural harmonics that extend the bandwidth seamlessly. Then apply
    # a spectral tilt to lift them to a natural level.
    #
    # Think of it as running the signal through a subtle tape machine.
    
    # Tape saturation: generates harmonics above the brick-wall cutoff
    VITALITY_SATURATION_DRIVE = 0.1   # 0..1: very subtle (0.1-0.2 recommended)
    VITALITY_SATURATION_KNEE_HZ = 9000 # Saturation increases above this freq
    VITALITY_SATURATION_MIX = 0.5     # 0..1: wet/dry blend of saturated signal
    
    # HF shelf: lifts the saturation-generated harmonics to audible level
    VITALITY_SHELF_GAIN_DB = 3.5       # Shelf boost in dB (less = less synthetic texture to fix)
    VITALITY_SHELF_FREQ_HZ = 15500     # Shelf center frequency (raised to avoid seam at 14k)
    
    # Spectral smoothing: removes any steps in the spectral envelope
    VITALITY_SMOOTH_STRENGTH = 0.8     # 0..1: how hard to enforce smooth envelope
    VITALITY_SMOOTH_WIDTH_OCT = 0.6    # Gaussian width in octaves
    
    # Warmth: gentle top-end rolloff for analog character
    VITALITY_WARMTH = 0.22             # 0..1: rolloff above 16kHz (lowered from 0.3)
    
    # Overall amount
    VITALITY_AMOUNT = 0.7              # 0..1: master wet/dry for entire vitality chain
    
    # Bandwidth detection
    HFR_CUTOFF_HZ = 0                  # 0 = auto-detect, >0 = override
    HFR_FRAME_LENGTH = 4096            # FFT size for analysis
    HFR_HOP_LENGTH = 1024

    # HF Naturalization — makes synthesized HF look/feel organic
    # NOTE: blur strength ramps with frequency (light near transition, heavier at Nyquist)
    NATURALIZE_TRANSITION_HZ = 15000.0   # Start naturalizing above this (Hz)
    NATURALIZE_DIFFUSION = 0.50          # 0..2: max blur at Nyquist; ramps from ~8% at transition
    NATURALIZE_NOISE_FLOOR_DB = -52.0    # dB: shaped noise floor to fill gaps
    NATURALIZE_PHASE_DIFFUSION = 0.20    # 0..1: how much to randomize HF phase

    # HF Seam Boost — focused lift at the bandwidth boundary (applied BEFORE naturalization)
    SEAM_BOOST_DB = 6.0                   # dB: peak boost at the seam (enough to fill the gap)
    SEAM_BOOST_HZ = 15000                # Forced to 15kHz
    SEAM_BOOST_ROLLOFF = 1.8             # dB/octave: gentler decay to sustain energy toward Nyquist
    SEAM_BOOST_ONSET_WIDTH = 3000.0      # Hz: wide ramp below seam for smooth blending

    # Dynamic Seam Bridge — replaces static seam boost with a per-frame
    # adaptive EQ bell that tracks the actual seam location and follows energy.
    DYNAMIC_SEAM_BRIDGE = True            # Enable (disables static seam boost)
    DYNAMIC_SEAM_DB = 7.0                 # Peak boost dB
    DYNAMIC_SEAM_Q = 9.0                  # Q factor of the bell
    DYNAMIC_SEAM_SEARCH_LO = 11000.0     # Hz: low end of seam search range
    DYNAMIC_SEAM_SEARCH_HI = 18000.0     # Hz: high end of seam search range
    DYNAMIC_SEAM_OFFSET_HZ = 600.0       # Hz: place boost center this far above detected seam
    DYNAMIC_SEAM_REF_LO = 11500.0        # Hz: energy reference band low
    DYNAMIC_SEAM_REF_HI = 14000.0        # Hz: energy reference band high
    DYNAMIC_SEAM_GAIN_MIN = 0.20         # Minimum gain multiplier (quiet frames)
    DYNAMIC_SEAM_GAIN_MAX = 1.40         # Maximum gain multiplier (loud frames)
    
    # =========================================================================
    
    # Griffin-Lim
    GL_ITERATIONS = 150
    GL_MOMENTUM = 0.99
    
    # Declipping
    DECLIP_ITERATIONS = 500
    DECLIP_LAMBDA = 0.08
    DECLIP_MULTI_RESOLUTION = True
    
    # Processing
    NUM_ENHANCEMENT_PASSES = 1
    PASS_DECAY_FACTOR = 0.5
    ANALYZE_FULL_FILE = True
    
    # GPU-specific
    GPU_BATCH_SIZE = 32
    USE_MIXED_PRECISION = False

    # Black-box ML stages (optional)
    ENABLE_APOLLO = True            # Apollo lossy-audio restoration (Stage 1)
    ENABLE_STEM_FIX = True          # Demucs stem-aware cleanup

    # =========================================================================
    # ORGANIZED DIRECTORY LAYOUT — everything next to the script
    # =========================================================================
    MODELS_DIR   = os.path.join(_SCRIPT_DIR, "models")
    CACHE_DIR    = os.path.join(_SCRIPT_DIR, "cache")
    OUTPUT_DIR   = os.path.join(_SCRIPT_DIR, "output")

    # Sub-directories for each model family
    APOLLO_DIR   = os.path.join(MODELS_DIR, "apollo")
    DEMUCS_DIR   = os.path.join(MODELS_DIR, "demucs")
    MSST_DIR     = os.path.join(MODELS_DIR, "msst")

    # =========================================================================
    # APOLLO — lossy MP3/AAC restoration via MSST framework
    # =========================================================================
    # Place your checkpoint + config inside  models/apollo/
    #   models/apollo/config_apollo.yaml
    #   models/apollo/pytorch_model.bin
    APOLLO_MODEL_TYPE = "apollo"
    APOLLO_CONFIG_PATH = os.path.join(APOLLO_DIR, "config_apollo.yaml")
    APOLLO_CHECKPOINT_PATH = os.path.join(APOLLO_DIR, "pytorch_model.bin")
    # Fallback download URLs (only used if local files are missing)
    APOLLO_CONFIG_URL = (
        "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training"
        "/refs/heads/main/configs/config_apollo.yaml"
    )
    APOLLO_CHECKPOINT_URL = (
        "https://huggingface.co/JusperLee/Apollo/resolve/main/pytorch_model.bin"
    )

    # Mix polish — VERY gentle (previous values caused "overloaded" sound)
    TRANSIENT_STRENGTH = 0.10       # Almost off — AI music doesn't need transient boost
    ROUGHNESS_REDUCTION = 0.40      # Gentle de-harsh only
    TRUEPEAK_TARGET_DBFS = -1.0

    # =========================================================================
    # OVERSAMPLING — run DSP chain at 2× for cleaner HF synthesis
    # =========================================================================
    OVERSAMPLE_FACTOR = 2            # 1 = off, 2 = 2× (default), 4 = 4×

    # =========================================================================
    # OUTPUT FORMAT — default to input's sr/bit depth (min 44.1k/16-bit)
    # =========================================================================
    OUTPUT_FORMAT = 'wav'            # 'wav', 'flac', 'mp3'
    OUTPUT_BIT_DEPTH = 0             # 0 = match input (min 16)
    OUTPUT_SAMPLE_RATE = 0           # 0 = match input (min 44100)

    # =========================================================================
    # MP-SENet — explicit magnitude/phase speech enhancement network
    # =========================================================================
    ENABLE_MPSENET = True            # Phase-aware restoration after HFPv2
    MPSENET_DIR = os.path.join(MODELS_DIR, "mpsenet")
    MPSENET_REPO_URL = "https://github.com/yxlu-0102/MP-SENet.git"

    # =========================================================================
    # SPECTRAL FLUX CORRECTION — restore transient sharpness in AI music
    # =========================================================================
    ENABLE_SPECTRAL_FLUX_CORRECTION = True
    SPECTRAL_FLUX_STRENGTH = 0.65    # 0..1: how aggressively to sharpen smeared attacks
    SPECTRAL_FLUX_THRESHOLD = 0.30   # Flux ratio below which a frame is "smeared"

    # =========================================================================
    # DEMUCS STEM-AWARE CLEANUP — v4 Hybrid Transformer model
    # =========================================================================
    # htdemucs_ft (fine-tuned) is slower but often cleaner than htdemucs.
    # Falls back to htdemucs if ft is unavailable.
    DEMUCS_MODEL_NAME = "htdemucs_ft"  # "htdemucs_ft" or "htdemucs"


# =========================
# GPU-Accelerated Declipping
# =========================

class GPUDeclipper:
    """GPU-accelerated A-SPADE declipping"""
    
    def __init__(self, config: UltimateGPUConfig):
        self.config = config
        self.iterations = config.DECLIP_ITERATIONS
        self.lambda_reg = config.DECLIP_LAMBDA
    
    def declip(self, audio: np.ndarray, sr: int,
               clipping_threshold: Optional[float] = None) -> np.ndarray:
        """Main declipping with GPU acceleration"""
        is_stereo = audio.ndim == 2
        
        if is_stereo:
            declipped = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                print(f"    Channel {ch+1}/{audio.shape[1]}")
                declipped[:, ch] = self._declip_gpu(audio[:, ch], sr, clipping_threshold)
            return declipped
        else:
            return self._declip_gpu(audio, sr, clipping_threshold)
    
    def _declip_gpu(self, audio: np.ndarray, sr: int,
                    clipping_threshold: Optional[float] = None) -> np.ndarray:
        """GPU-accelerated declipping for mono channel"""
        if clipping_threshold is None:
            clipping_threshold = self._estimate_threshold(audio)
        
        print(f"      Threshold: {clipping_threshold:.4f}")
        
        # Create masks
        reliable_mask = np.abs(audio) < clipping_threshold
        clipped_high = audio >= clipping_threshold
        clipped_low = audio <= -clipping_threshold
        
        if not np.any(clipped_high) and not np.any(clipped_low):
            return audio
        
        # Initialize
        frame_length = 2048
        hop_length = 512
        stft_engine = GPUStft(n_fft=frame_length, hop_length=hop_length)
        
        x_rec = audio.copy()
        
        # Move masks to GPU if using CuPy
        if BACKEND == "cupy" and cp is not None:
            reliable_gpu = cp.asarray(reliable_mask)
            high_gpu = cp.asarray(clipped_high)
            low_gpu = cp.asarray(clipped_low)
            audio_gpu = cp.asarray(audio)
        
        for iteration in range(self.iterations):
            # Apply consistency constraint
            if BACKEND == "cupy" and cp is not None:
                x_rec_gpu = cp.asarray(x_rec)
                x_rec_gpu = self._apply_consistency_gpu(
                    x_rec_gpu, audio_gpu, reliable_gpu, high_gpu, low_gpu, clipping_threshold
                )
                x_rec = cp.asnumpy(x_rec_gpu)
            else:
                x_rec = self._apply_consistency_cpu(
                    x_rec, audio, reliable_mask, clipped_high, clipped_low, clipping_threshold
                )
            
            # STFT on GPU
            D = stft_engine.stft(x_rec)
            D_magnitude = np.abs(D)
            D_phase = np.angle(D)
            
            # Soft thresholding
            progress = iteration / self.iterations
            adaptive_lambda = self.lambda_reg * (1 - 0.5 * progress)
            threshold = adaptive_lambda * np.median(D_magnitude)
            
            if BACKEND == "cupy" and cp is not None:
                D_mag_gpu = cp.asarray(D_magnitude)
                D_mag_sparse = cp.maximum(D_mag_gpu - threshold, 0)
                D_magnitude_sparse = cp.asnumpy(D_mag_sparse)
            else:
                D_magnitude_sparse = np.maximum(D_magnitude - threshold, 0)
            
            D_sparse = D_magnitude_sparse * np.exp(1j * D_phase)
            
            # ISTFT on GPU
            x_rec = stft_engine.istft(D_sparse, length=len(audio))
            
            if (iteration + 1) % 100 == 0:
                print(f"        Iteration {iteration + 1}/{self.iterations}")
        
        # Final consistency
        if BACKEND == "cupy" and cp is not None:
            x_rec_gpu = cp.asarray(x_rec)
            x_rec_gpu = self._apply_consistency_gpu(
                x_rec_gpu, audio_gpu, reliable_gpu, high_gpu, low_gpu, clipping_threshold
            )
            x_rec = cp.asnumpy(x_rec_gpu)
        else:
            x_rec = self._apply_consistency_cpu(
                x_rec, audio, reliable_mask, clipped_high, clipped_low, clipping_threshold
            )
        
        return x_rec
    
    def _apply_consistency_gpu(self, x_rec, audio, reliable, high, low, threshold):
        """GPU-accelerated consistency constraint"""
        # Use CuPy's where for GPU-accelerated conditional assignment
        result = cp.where(reliable, audio, x_rec)
        result = cp.where(high & (result < threshold), threshold * 1.05, result)
        result = cp.where(low & (result > -threshold), -threshold * 1.05, result)
        return result
    
    @staticmethod
    @jit(nopython=True, cache=True, parallel=True)
    def _apply_consistency_cpu(x_rec, audio, reliable, high, low, threshold):
        """CPU fallback for consistency constraint"""
        result = x_rec.copy()
        for i in prange(len(x_rec)):
            if reliable[i]:
                result[i] = audio[i]
            elif high[i]:
                if result[i] < threshold:
                    result[i] = threshold * 1.05
            elif low[i]:
                if result[i] > -threshold:
                    result[i] = -threshold * 1.05
        return result
    
    @staticmethod
    def _estimate_threshold(audio: np.ndarray) -> float:
        audio_abs = np.abs(audio)
        max_val = np.max(audio_abs)
        
        hist, bin_edges = np.histogram(audio_abs, bins=500)
        peak_threshold = 0.8 * max_val
        high_bins = bin_edges[:-1] > peak_threshold
        
        if np.any(high_bins):
            high_hist = hist[high_bins]
            if np.max(high_hist) > 10 * np.median(hist):
                threshold_idx = np.where(high_bins)[0][0]
                return bin_edges[threshold_idx]
        
        return np.percentile(audio_abs, 99.5)


# =========================
# GPU-Accelerated DSRE
# =========================

def freq_shift_gpu(x: np.ndarray, f_shift: float, sr: float) -> np.ndarray:
    """GPU-accelerated frequency shifting"""
    if len(x) == 0:
        return x
    
    N_orig = len(x)
    N_padded = 1 << int(np.ceil(np.log2(max(1, N_orig * 2))))
    
    # FFT on GPU
    if BACKEND == "cupy" and cp is not None:
        x_gpu = cp.asarray(x)
        X = cp.fft.fft(x_gpu, n=N_padded)
        
        h = cp.zeros(N_padded)
        h[0] = 1
        h[1:N_padded//2] = 2
        h[N_padded//2] = 1
        
        analytic = cp.fft.ifft(X * h)
        
        # Phase shift
        phase_factors = cp.exp(2j * cp.pi * f_shift * cp.arange(N_orig) / sr)
        shifted = analytic[:N_orig] * phase_factors
        
        return cp.asnumpy(shifted.real)
    
    elif BACKEND == "torch" and torch is not None:
        x_gpu = torch.from_numpy(np.ascontiguousarray(x)).cuda().to(torch.complex128)
        X = torch.fft.fft(x_gpu, n=N_padded)
        
        h = torch.zeros(N_padded, dtype=torch.complex128, device='cuda')
        h[0] = 1
        h[1:N_padded//2] = 2
        h[N_padded//2] = 1
        
        analytic = torch.fft.ifft(X * h)
        
        t = torch.arange(N_orig, device='cuda', dtype=torch.float64)
        phase_factors = torch.exp(2j * np.pi * f_shift * t / sr)
        shifted = analytic[:N_orig] * phase_factors
        
        return shifted.real.cpu().numpy()
    
    else:
        # CPU fallback
        X = np.fft.fft(x, n=N_padded)
        h = np.zeros(N_padded)
        h[0] = 1
        h[1:N_padded//2] = 2
        h[N_padded//2] = 1
        analytic = np.fft.ifft(X * h)
        
        phase_factors = np.exp(2j * np.pi * f_shift * np.arange(N_orig) / sr)
        shifted = analytic[:N_orig] * phase_factors
        
        return shifted.real


def zansei_gpu(
    x: np.ndarray,
    sr: int,
    m: int = 16,
    decay: float = 1.15,
    pre_hp: float = 2500.0,
    post_hp: float = 14000.0,
    filter_order: int = 15,
    num_passes: int = 2,
    pass_decay: float = 0.7,
) -> np.ndarray:
    """GPU-accelerated DSRE"""
    is_mono = x.ndim == 1
    if is_mono:
        x = x[np.newaxis, :]
    
    result = x.copy()
    
    for pass_num in range(num_passes):
        pass_factor = pass_decay ** pass_num
        print(f"      DSRE Pass {pass_num + 1}/{num_passes}")
        result = _zansei_single_pass_gpu(
            result, sr, m, decay * pass_factor, pre_hp, post_hp, filter_order
        )
    
    return result[0] if is_mono else result


def _zansei_single_pass_gpu(
    x: np.ndarray,
    sr: int,
    m: int,
    decay: float,
    pre_hp: float,
    post_hp: float,
    filter_order: int,
) -> np.ndarray:
    """Single pass of GPU-accelerated DSRE with smooth frequency blending"""
    channels, n_samples = x.shape
    
    # Store original peak for later normalization
    original_peak = np.max(np.abs(x))
    
    # Pre-filter (CPU - small overhead)
    if 0 < pre_hp < sr * 0.5:
        sos = signal.butter(filter_order, pre_hp / (sr / 2.0), "highpass", output='sos')
        d_src = signal.sosfiltfilt(sos, x, axis=1)
    else:
        d_src = x
    
    # Calculate shift frequencies
    shift_freqs = sr * np.arange(1, m + 1) / (m * 2.0)
    decay_factors = np.exp(-np.arange(1, m + 1) ** 0.8 * decay / m)
    
    # Process shifts in parallel on GPU
    d_res = np.zeros_like(x)
    
    for i in range(m):
        for ch in range(channels):
            shifted = freq_shift_gpu(d_src[ch], shift_freqs[i], sr)
            d_res[ch] += shifted * decay_factors[i]
    
    # Post-filter with SMOOTH CROSSFADE instead of hard highpass
    # This prevents the 14kHz notch issue
    if 0 < post_hp < sr * 0.5:
        # Create a smooth highpass using frequency-domain crossfade
        d_res = _smooth_highpass_crossfade(d_res, sr, post_hp, filter_order)
    
    # Spectral smoothing: DSRE frequency shifts create discrete tonal copies.
    try:
        _dsre_nfft = 2048
        _dsre_hop = 512
        _dsre_stft = GPUStft(n_fft=_dsre_nfft, hop_length=_dsre_hop)
        for _dch in range(channels):
            _S_d = _dsre_stft.stft(d_res[_dch].astype(np.float64))
            _mag_d = np.abs(_S_d)
            _pha_d = np.angle(_S_d)
            for _ in range(2):
                _mag_d = ndimage.gaussian_filter1d(
                    _mag_d.astype(np.float32), sigma=20.0, axis=0, mode='nearest'
                ).astype(np.float64)
            _S_d_smooth = _mag_d * np.exp(1j * _pha_d)
            d_res[_dch] = _dsre_stft.istft(_S_d_smooth, length=n_samples)
    except Exception:
        pass
    
    # FIXED: Energy-envelope-modulated mixing.
    # Instead of a flat 15% mix, modulate the enhancement by the signal's
    # own energy envelope so quiet passages get proportionally less HF injection
    # and attacks get the full amount.
    enhancement_ratio = 0.15  # Peak enhancement adds ~15% of original energy
    d_res_peak = np.max(np.abs(d_res)) + 1e-12
    d_res_scaled = d_res * (original_peak * enhancement_ratio / d_res_peak)

    # Compute a fast energy envelope from the original signal (per-sample)
    # Use short window (~5ms) for attack sensitivity + longer (~30ms) for body
    env_short_win = max(16, int(sr * 0.005))
    env_long_win = max(64, int(sr * 0.030))
    x_mono = np.mean(np.abs(x), axis=0) if channels > 1 else np.abs(x[0])
    try:
        env_short = ndimage.maximum_filter1d(x_mono, size=env_short_win, mode='nearest')
        env_long = ndimage.uniform_filter1d(x_mono ** 2, size=env_long_win, mode='nearest')
        env_long = np.sqrt(env_long + 1e-12)
        # Use the faster of the two envelopes (preserves attacks)
        env = np.maximum(env_short, env_long)
    except Exception:
        env = ndimage.uniform_filter1d(x_mono, size=env_long_win, mode='nearest')
    env_peak = np.max(env) + 1e-12
    env_ratio = np.clip(env / env_peak, 0.0, 1.0)  # 0=silence, 1=peak
    # Apply power curve so quiet parts drop faster than linear
    env_ratio = env_ratio ** 1.5

    # Modulate DSRE contribution by energy envelope
    for _ch in range(channels):
        d_res_scaled[_ch] *= env_ratio

    # Mix and normalize to original peak
    y = x + d_res_scaled
    y_peak = np.max(np.abs(y))
    if y_peak > original_peak:
        y = y * (original_peak / y_peak)
    
    return y


def _smooth_highpass_crossfade(audio: np.ndarray, sr: int, cutoff_hz: float,
                               order: int = 6) -> np.ndarray:
    """Highpass via STFT with an equal-power transition that reaches unity AT the cutoff.

    Why: this helper is used to isolate HF components for additive stages.
    A transition that reaches unity only *above* cutoff can show up as a visible
    "step" (power jump) at the end of the transition band (e.g. ~14.5 kHz when
    cutoff is ~12 kHz). We want the transition to finish right at cutoff.

    Behavior:
      - 0.0 gain well below the cutoff
      - raised-cosine ramp that ends at cutoff_bin-1
      - 1.0 gain from cutoff_bin upward
    """
    channels, n_samples = audio.shape

    n_fft = 2048
    hop_length = 512
    stft_engine = GPUStft(n_fft=n_fft, hop_length=hop_length)

    result = np.zeros_like(audio)

    for ch in range(channels):
        spec = stft_engine.stft(audio[ch])
        num_bins = spec.shape[0]

        freq_per_bin = (sr / 2.0) / num_bins
        cutoff_bin = int(cutoff_hz / (freq_per_bin + 1e-12))
        cutoff_bin = int(np.clip(cutoff_bin, 8, num_bins - 8))

        # Wide transition BELOW the cutoff only (ends at cutoff)
        transition_width = max(24, int(cutoff_bin * 0.25))
        fade_start = max(0, cutoff_bin - transition_width)
        fade_end = cutoff_bin  # unity from cutoff_bin upward

        weights = np.zeros(num_bins, dtype=np.float32)
        weights[fade_end:] = 1.0

        if fade_end > fade_start:
            t = np.linspace(0.0, 1.0, fade_end - fade_start, dtype=np.float32)
            ramp = 0.5 - 0.5 * np.cos(np.pi * t)  # raised cosine
            weights[fade_start:fade_end] = ramp

        spec_filtered = spec * weights[:, np.newaxis]
        
        # Gentle temporal smoothing to prevent frame-boundary seams (but not smear transients)
        try:
            mag_f = np.abs(spec_filtered)
            phase_f = np.angle(spec_filtered)
            mag_f = ndimage.gaussian_filter1d(
                mag_f.astype(np.float32), sigma=1.5, axis=1, mode='nearest'
            ).astype(np.float64)
            spec_filtered = mag_f * np.exp(1j * phase_f)
        except Exception:
            pass
        
        result[ch] = stft_engine.istft(spec_filtered, length=n_samples)

    return result


# =========================
# GPU-Accelerated Griffin-Lim
# =========================

class GPUGriffinLim:
    """GPU-accelerated Griffin-Lim algorithm"""
    
    def __init__(self, n_fft: int = 2048, hop_length: int = 512,
                 n_iter: int = 150, momentum: float = 0.99):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_iter = n_iter
        self.momentum = momentum
        self.stft_engine = GPUStft(n_fft=n_fft, hop_length=hop_length)
    
    def reconstruct(self, magnitude: np.ndarray, 
                    length: Optional[int] = None) -> np.ndarray:
        """Reconstruct audio from magnitude spectrogram"""
        if BACKEND == "cupy" and cp is not None:
            return self._griffin_lim_cupy(magnitude, length)
        elif BACKEND == "torch" and torch is not None:
            return self._griffin_lim_torch(magnitude, length)
        else:
            return self._griffin_lim_cpu(magnitude, length)
    
    def _griffin_lim_cupy(self, magnitude: np.ndarray, 
                          length: Optional[int]) -> np.ndarray:
        """CuPy-accelerated Griffin-Lim"""
        mag_gpu = cp.asarray(magnitude)
        
        # Random phase initialization
        phase = cp.exp(2j * cp.pi * cp.random.rand(*magnitude.shape))
        spec = mag_gpu * phase
        
        momentum_buffer = cp.zeros_like(spec)
        prev_spec = spec.copy()
        
        for i in range(self.n_iter):
            # ISTFT
            audio = self._istft_cupy(spec, length)
            
            # STFT
            new_spec = self._stft_cupy(audio)
            
            # Ensure same shape
            if new_spec.shape != mag_gpu.shape:
                if new_spec.shape[1] < mag_gpu.shape[1]:
                    new_spec = cp.pad(new_spec, ((0, 0), (0, mag_gpu.shape[1] - new_spec.shape[1])))
                else:
                    new_spec = new_spec[:, :mag_gpu.shape[1]]
            
            # Apply magnitude constraint
            phase = new_spec / (cp.abs(new_spec) + 1e-8)
            spec_updated = mag_gpu * phase
            
            # Momentum
            if i > 0:
                adaptive_momentum = self.momentum * (1 - 1/(i + 1))
                momentum_buffer = adaptive_momentum * momentum_buffer + (spec_updated - prev_spec)
                spec = spec_updated + adaptive_momentum * momentum_buffer
                spec = mag_gpu * (spec / (cp.abs(spec) + 1e-8))
            else:
                spec = spec_updated
            
            prev_spec = spec.copy()
            
            if (i + 1) % 50 == 0:
                print(f"        GL iteration {i + 1}/{self.n_iter}")
        
        audio = self._istft_cupy(spec, length)
        return cp.asnumpy(audio)
    
    def _stft_cupy(self, x):
        """CuPy STFT"""
        window = cp.asarray(signal.get_window('hann', self.n_fft))
        n_frames = 1 + (len(x) - self.n_fft) // self.hop_length
        
        indices = cp.arange(self.n_fft)[None, :] + cp.arange(n_frames)[:, None] * self.hop_length
        indices = cp.minimum(indices, len(x) - 1)
        frames = x[indices] * window
        
        return cp.fft.rfft(frames, axis=1).T
    
    def _istft_cupy(self, X, length):
        """CuPy ISTFT"""
        window = cp.asarray(signal.get_window('hann', self.n_fft))
        n_frames = X.shape[1]
        
        if length is None:
            length = (n_frames - 1) * self.hop_length + self.n_fft
        
        frames = cp.fft.irfft(X.T, n=self.n_fft, axis=1)
        frames = frames * window
        
        output = cp.zeros(length, dtype=cp.float64)
        window_sum = cp.zeros(length, dtype=cp.float64)
        
        for i in range(n_frames):
            start = i * self.hop_length
            end = min(start + self.n_fft, length)
            frame_len = end - start
            output[start:end] += frames[i, :frame_len]
            window_sum[start:end] += window[:frame_len] ** 2
        
        window_sum = cp.maximum(window_sum, 1e-8)
        return output / window_sum
    
    def _griffin_lim_torch(self, magnitude: np.ndarray,
                           length: Optional[int]) -> np.ndarray:
        """PyTorch-accelerated Griffin-Lim"""
        mag_gpu = torch.from_numpy(np.ascontiguousarray(magnitude)).cuda()
        
        # Random phase
        phase = torch.exp(2j * np.pi * torch.rand(*magnitude.shape, device='cuda'))
        spec = mag_gpu * phase
        
        window = torch.from_numpy(
            signal.get_window('hann', self.n_fft).astype(np.float32)
        ).cuda()
        
        for i in range(self.n_iter):
            audio = torch.istft(
                spec, n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.n_fft, window=window, length=length
            )
            
            new_spec = torch.stft(
                audio, n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.n_fft, window=window, return_complex=True
            )
            
            if new_spec.shape != mag_gpu.shape:
                if new_spec.shape[1] < mag_gpu.shape[1]:
                    new_spec = torch.nn.functional.pad(
                        new_spec, (0, mag_gpu.shape[1] - new_spec.shape[1])
                    )
                else:
                    new_spec = new_spec[:, :mag_gpu.shape[1]]
            
            phase = new_spec / (torch.abs(new_spec) + 1e-8)
            spec = mag_gpu * phase
            
            if (i + 1) % 50 == 0:
                print(f"        GL iteration {i + 1}/{self.n_iter}")
        
        audio = torch.istft(
            spec, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=window, length=length
        )
        return audio.cpu().numpy()
    
    def _griffin_lim_cpu(self, magnitude: np.ndarray,
                         length: Optional[int]) -> np.ndarray:
        """CPU fallback Griffin-Lim"""
        phase = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
        spec = magnitude * phase
        
        for i in range(self.n_iter):
            audio = self.stft_engine.istft(spec, length)
            new_spec = self.stft_engine.stft(audio)
            
            if new_spec.shape != magnitude.shape:
                if new_spec.shape[1] < magnitude.shape[1]:
                    new_spec = np.pad(new_spec, ((0, 0), (0, magnitude.shape[1] - new_spec.shape[1])))
                else:
                    new_spec = new_spec[:, :magnitude.shape[1]]
            
            phase = new_spec / (np.abs(new_spec) + 1e-8)
            spec = magnitude * phase
            
            if (i + 1) % 50 == 0:
                print(f"        GL iteration {i + 1}/{self.n_iter}")
        
        return self.stft_engine.istft(spec, length)


# =========================
# ANALOG VITALITY — Seamless HF Extension via Saturation + Shelf + Smoothing
# =========================

class AnalogVitality:
    """Extend bandwidth and add life to AI-generated music.

    PHILOSOPHY: Previous HFR approaches (SBR, harmonic transposition, noise shaping)
    all failed because they SYNTHESIZE content above the cutoff that has a fundamentally
    different texture than the original below it. Any crossfade/blend at the boundary
    creates a visible spectrogram seam and audible artifact.

    New approach: DON'T SYNTHESIZE. Instead, use the same physics that makes analog
    recordings have natural high-frequency content:

    1. TAPE SATURATION — Gentle nonlinear waveshaping creates harmonics FROM the
       existing signal. A 7 kHz tone naturally generates 14, 21 kHz harmonics.
       Content that was brick-walled at 14 kHz now naturally extends above it.
       No seam possible because the new content IS the old content's harmonics.

    2. HF SHELF EQ — The saturation harmonics are quiet. A smooth shelf boost above
       ~12 kHz lifts them to a natural level. This is a smooth filter, not a step.

    3. SPECTRAL ENVELOPE SMOOTHING — Safety net: measures the full spectral shape
       and gently corrects any remaining steps from upstream processing (AudioSR,
       etc.). Works on the entire spectrum, not just around a cutoff.

    4. WARMTH FILTER — Gentle rolloff above 16 kHz for analog character, preventing
       the saturated harmonics from sounding harsh or brittle.

    The result: seamless, warm, natural-sounding high frequency extension.
    """

    def __init__(self, config: UltimateGPUConfig):
        self.config = config
        self.stft_engine = GPUStft(
            n_fft=config.HFR_FRAME_LENGTH,
            hop_length=config.HFR_HOP_LENGTH
        )

    def restore(self, audio: np.ndarray, sr: int,
                lowpass_hz: Optional[int] = None) -> np.ndarray:
        """Apply the full vitality chain to stereo audio."""
        original_peak = np.max(np.abs(audio)) + 1e-12
        cfg = self.config

        # Detect bandwidth
        if audio.ndim == 2:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio
        if lowpass_hz is None or lowpass_hz <= 0:
            lowpass_hz = self._detect_bandwidth(mono, sr)

        print(f"      Detected bandwidth: ~{lowpass_hz:.0f} Hz")
        print(f"      Saturation drive={cfg.VITALITY_SATURATION_DRIVE:.2f}, "
              f"shelf={cfg.VITALITY_SHELF_GAIN_DB:.1f}dB@{cfg.VITALITY_SHELF_FREQ_HZ}Hz")

        amount = float(cfg.VITALITY_AMOUNT)
        dry = audio.copy()

        # ----- STEP 1: Multiband Tape Saturation -----
        audio = self._tape_saturate(audio, sr, lowpass_hz)

        # ----- STEP 2: HF Shelf EQ -----
        audio = self._hf_shelf(audio, sr)

        # ----- STEP 3: Warmth filter -----
        audio = self._warmth_filter(audio, sr)

        # ----- STEP 4: Spectral envelope smoothing -----
        audio = self._spectral_smooth(audio, sr)

        # ----- Master wet/dry -----
        audio = dry * (1.0 - amount) + audio * amount

        # Preserve peak
        out_peak = np.max(np.abs(audio)) + 1e-12
        if out_peak > original_peak:
            audio *= (original_peak / out_peak)

        return audio

    def _detect_bandwidth(self, audio: np.ndarray, sr: int) -> int:
        """Detect where the audio's energy drops off sharply."""
        eps = 1e-12
        S = self.stft_engine.stft(audio)
        mag = np.abs(S)
        e = np.mean(mag, axis=1) + eps

        try:
            e_s = ndimage.gaussian_filter1d(e.astype(np.float32), sigma=2.0, mode="nearest")
        except Exception:
            e_s = e

        e_db = 20.0 * np.log10(e_s + eps)
        peak_db = float(np.max(e_db))

        n_fft = self.config.HFR_FRAME_LENGTH
        bin_hz = sr / float(n_fft)

        min_hz = max(2000.0, getattr(self.config, '_detected_bandwidth', 15000.0) * 0.50)
        max_hz = max(min_hz + 1000.0, (sr / 2.0) - 1500.0)

        min_bin = int(min_hz / bin_hz)
        max_bin = int(max_hz / bin_hz)
        min_bin = max(16, min(min_bin, len(e_db) - 64))
        max_bin = max(min_bin + 32, min(max_bin, len(e_db) - 16))

        drop_db = 34.0
        win = 48
        cutoff = None
        for b in range(min_bin, max_bin):
            tail = e_db[b:min(len(e_db), b + win)]
            if tail.size < win // 2:
                break
            if np.median(tail) < (peak_db - drop_db):
                cutoff = b
                break

        if cutoff is None:
            de = np.diff(e_db)
            region = de[min_bin:max_bin]
            if region.size > 0:
                cutoff = int(np.argmin(region) + min_bin)
            else:
                cutoff = int(self.config.HFR_FRAME_LENGTH // 4)

        cutoff = int(np.clip(cutoff, 48, (self.config.HFR_FRAME_LENGTH // 2) - 8))
        return int(cutoff * sr / self.config.HFR_FRAME_LENGTH)

    # -----------------------------------------------------------------
    # STEP 1: Tape saturation (frequency-dependent)
    # -----------------------------------------------------------------
    def _tape_saturate(self, audio: np.ndarray, sr: int, cutoff_hz: int) -> np.ndarray:
        """Apply gentle, frequency-dependent tape saturation.

        The saturation drive increases with frequency. Below the knee frequency,
        almost no saturation. Above it, increasing drive. This means:
        - Low/mid frequencies: untouched (preserve tonal character)
        - High frequencies: gentle harmonics generated
        - Above the cutoff: harmonics extend the bandwidth naturally

        Uses tanh() soft clipping — creates mostly odd harmonics with a warm character.
        We also add a small asymmetric component for even harmonics (2nd harmonic warmth).
        """
        cfg = self.config
        drive = float(cfg.VITALITY_SATURATION_DRIVE)
        knee_hz = float(cfg.VITALITY_SATURATION_KNEE_HZ)
        mix = float(cfg.VITALITY_SATURATION_MIX)

        if drive < 0.01 or mix < 0.01:
            return audio

        n_fft = cfg.HFR_FRAME_LENGTH
        hop = cfg.HFR_HOP_LENGTH
        result = audio.copy()

        for ch in range(audio.shape[1] if audio.ndim == 2 else 1):
            x = audio[:, ch] if audio.ndim == 2 else audio

            S = self.stft_engine.stft(x)
            num_bins, num_frames = S.shape
            bin_hz = sr / float(n_fft)
            eps = 1e-10

            # Build frequency-dependent drive curve
            freqs = np.arange(num_bins) * bin_hz
            # Drive ramps from 0 below knee to full above cutoff
            drive_curve = np.zeros(num_bins, dtype=np.float64)
            for b in range(num_bins):
                if freqs[b] <= knee_hz:
                    drive_curve[b] = 0.0  # No saturation below knee
                elif freqs[b] >= cutoff_hz:
                    drive_curve[b] = drive  # Full drive at/above cutoff
                else:
                    # Smooth ramp between knee and cutoff
                    t = (freqs[b] - knee_hz) / max(1.0, cutoff_hz - knee_hz)
                    drive_curve[b] = drive * (t * t)  # Quadratic ramp

            # Apply saturation per frequency band in STFT domain
            # We do this by converting back to time, applying nonlinearity,
            # and converting back. But per-bin saturation is better done
            # in a multiband fashion.
            #
            # Simpler and more effective: apply saturation in time domain
            # to frequency-filtered versions of the signal.

            # Split signal into low (below knee, clean) and high (above knee, saturated)
            pass  # We'll do this differently — see below

        # More efficient approach: bandpass → saturate → add back
        # This avoids per-bin STFT manipulation and sounds more natural

        for ch in range(audio.shape[1] if audio.ndim == 2 else 1):
            x = audio[:, ch] if audio.ndim == 2 else audio
            x = x.astype(np.float64)

            # Extract high-frequency content (above knee)
            try:
                sos_hp = signal.butter(4, knee_hz, btype='highpass', fs=sr, output='sos')
                x_hi = signal.sosfiltfilt(sos_hp, x).astype(np.float64)
            except Exception:
                x_hi = x.copy()

            # Normalize HF band for consistent saturation behavior
            hi_peak = np.max(np.abs(x_hi)) + 1e-12
            x_hi_norm = x_hi / hi_peak

            # Apply tape saturation (tanh soft clip)
            # drive controls how hard we push into the nonlinearity
            # Effective drive = drive * 3..8 (tanh needs larger values to be audible)
            effective_drive = 3.0 + drive * 12.0  # 0.15 → ~4.8

            saturated = np.tanh(effective_drive * x_hi_norm) / np.tanh(effective_drive)

            # Add subtle 2nd harmonic (even harmonic warmth via asymmetric clipping)
            # x + k*x^2 adds 2nd harmonic
            second_harmonic = 0.15 * drive * x_hi_norm * np.abs(x_hi_norm)
            saturated = saturated + second_harmonic
            # Re-normalize
            sat_peak = np.max(np.abs(saturated)) + 1e-12
            saturated = saturated / sat_peak * hi_peak

            # The saturation-generated content is: saturated - x_hi (the difference)
            harmonic_content = saturated - x_hi

            # --- Noise-modulate the harmonics so they aren't perfectly clean lines ---
            # Real analog harmonics are amplitude-modulated by circuit noise.
            # A subtle, slow noise envelope breaks up the spectrogram "laser lines".
            noise_mod_depth = 0.7   # 0..1: strong amplitude flutter
            mod_env_hz = 50.0       # modulation rate (faster flutter = more breakup)
            t_sec = np.arange(len(harmonic_content), dtype=np.float64) / sr
            # Band-limited noise envelope (smooth random AM)
            np.random.seed(42 + ch)
            raw_noise = np.random.randn(len(harmonic_content))
            # Lowpass the noise to mod_env_hz
            try:
                sos_mod = signal.butter(2, mod_env_hz / (sr / 2.0), 'lowpass', output='sos')
                mod_envelope = signal.sosfiltfilt(sos_mod, raw_noise)
                mod_envelope = mod_envelope / (np.max(np.abs(mod_envelope)) + 1e-12)
                harmonic_content = harmonic_content * (1.0 + noise_mod_depth * mod_envelope)
            except Exception:
                pass  # If filter fails, skip modulation

            # --- Spectral smearing: blur harmonic lines in frequency domain ---
            # The tanh saturation creates discrete harmonic lines (3x, 5x, 7x).
            # Smearing them in the frequency axis makes them look/sound like
            # natural broadband harmonics instead of synthetic "laser lines".
            try:
                _smear_nfft = n_fft
                _smear_hop = hop
                _smear_win = np.hanning(_smear_nfft).astype(np.float64)

                # Pad and STFT the harmonic content
                _hc_padded = np.pad(harmonic_content, (_smear_nfft, _smear_nfft), mode='constant')
                _n_frames_hc = 1 + (len(_hc_padded) - _smear_nfft) // _smear_hop
                _S_hc = np.zeros((_smear_nfft // 2 + 1, _n_frames_hc), dtype=np.complex128)
                for _fi in range(_n_frames_hc):
                    _start = _fi * _smear_hop
                    _frame = _hc_padded[_start:_start + _smear_nfft] * _smear_win
                    _S_hc[:, _fi] = np.fft.rfft(_frame)

                _mag_hc = np.abs(_S_hc)
                _phase_hc = np.angle(_S_hc)

                # Gaussian blur along frequency axis — sigma=30 bins, 3 passes
                _mag_smeared = _mag_hc.copy()
                for _smear_pass in range(3):
                    _mag_smeared = ndimage.gaussian_filter1d(
                        _mag_smeared.astype(np.float32), sigma=30.0, axis=0, mode='nearest'
                    ).astype(np.float64)
                # Gentle temporal blur to break up time-consistent lines
                _mag_smeared = ndimage.gaussian_filter1d(
                    _mag_smeared.astype(np.float32), sigma=2.0, axis=1, mode='nearest'
                ).astype(np.float64)

                # Preserve total energy per frame
                _frame_energy_orig = np.sum(_mag_hc, axis=0) + 1e-12
                _frame_energy_smeared = np.sum(_mag_smeared, axis=0) + 1e-12
                _mag_smeared *= (_frame_energy_orig / _frame_energy_smeared)[None, :]

                # Reconstruct with original phase
                _S_smeared = _mag_smeared * np.exp(1j * _phase_hc)
                _out_len = (_n_frames_hc - 1) * _smear_hop + _smear_nfft
                _hc_rebuilt = np.zeros(_out_len, dtype=np.float64)
                _hc_window_sum = np.zeros(_out_len, dtype=np.float64)
                for _fi in range(_n_frames_hc):
                    _start = _fi * _smear_hop
                    _frame_out = np.fft.irfft(_S_smeared[:, _fi], n=_smear_nfft) * _smear_win
                    _hc_rebuilt[_start:_start + _smear_nfft] += _frame_out
                    _hc_window_sum[_start:_start + _smear_nfft] += _smear_win ** 2

                _hc_window_sum = np.maximum(_hc_window_sum, 1e-8)
                _hc_rebuilt /= _hc_window_sum

                # Trim padding
                harmonic_content = _hc_rebuilt[_smear_nfft:_smear_nfft + len(harmonic_content)]
            except Exception:
                pass  # If smearing fails, use the noise-modulated version

            # Mix the harmonic content back in
            if audio.ndim == 2:
                result[:, ch] = x + harmonic_content * mix
            else:
                result = x + harmonic_content * mix

        return result

    # -----------------------------------------------------------------
    # STEP 2: High-frequency shelf EQ
    # -----------------------------------------------------------------
    def _hf_shelf(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply a smooth high-shelf boost to lift the generated harmonics.

        Uses a 2nd-order shelf filter — very smooth frequency response,
        no sharp transitions that could create seams.
        """
        cfg = self.config
        gain_db = float(cfg.VITALITY_SHELF_GAIN_DB)
        freq_hz = float(cfg.VITALITY_SHELF_FREQ_HZ)

        if abs(gain_db) < 0.1:
            return audio

        # Design high-shelf biquad
        # Using cookbook formula for high shelf
        A = 10.0 ** (gain_db / 40.0)  # sqrt of linear gain
        w0 = 2.0 * np.pi * freq_hz / sr
        alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)  # Q = 0.707 (Butterworth)

        cos_w0 = np.cos(w0)

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

        # Normalize
        b = np.array([b0 / a0, b1 / a0, b2 / a0])
        a = np.array([1.0, a1 / a0, a2 / a0])

        # Apply with zero-phase (filtfilt) for no phase distortion
        result = audio.copy()
        for ch in range(audio.shape[1] if audio.ndim == 2 else 1):
            x = audio[:, ch] if audio.ndim == 2 else audio
            try:
                if audio.ndim == 2:
                    result[:, ch] = signal.filtfilt(b, a, x).astype(audio.dtype)
                else:
                    result = signal.filtfilt(b, a, x).astype(audio.dtype)
            except Exception:
                pass

        return result

    # -----------------------------------------------------------------
    # STEP 3: Warmth filter (gentle HF rolloff)
    # -----------------------------------------------------------------
    def _warmth_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Gentle lowpass rolloff above 16 kHz for analog warmth.

        This tames any harshness from the saturation harmonics and gives
        the overall sound a warmer, more analog character.
        Uses a 1st-order lowpass for the gentlest possible slope (~6 dB/oct).
        """
        warmth = float(self.config.VITALITY_WARMTH)
        if warmth < 0.01:
            return audio

        # Effective cutoff: warmth=0 → no filter, warmth=1 → 12kHz cutoff
        # Interpolate between 20kHz (barely audible) and 12kHz (warm)
        cutoff_hz = 20000.0 - warmth * 8000.0  # 0→20kHz, 1→12kHz
        cutoff_hz = max(10000.0, min(cutoff_hz, sr / 2.0 - 1000.0))

        try:
            # 1st order = very gentle slope (6 dB/oct)
            sos = signal.butter(1, cutoff_hz, btype='lowpass', fs=sr, output='sos')
            result = audio.copy()
            for ch in range(audio.shape[1] if audio.ndim == 2 else 1):
                x = audio[:, ch] if audio.ndim == 2 else audio
                filtered = signal.sosfiltfilt(sos, x).astype(audio.dtype)
                # Blend: only partially apply the filter
                if audio.ndim == 2:
                    result[:, ch] = audio[:, ch] * (1.0 - warmth * 0.5) + filtered * (warmth * 0.5)
                else:
                    result = audio * (1.0 - warmth * 0.5) + filtered * (warmth * 0.5)
            return result
        except Exception:
            return audio

    # -----------------------------------------------------------------
    # STEP 4: Spectral envelope smoothing
    # -----------------------------------------------------------------
    def _spectral_smooth(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Smooth the spectral envelope to remove any remaining steps.

        Measures the full spectral shape, computes a heavily-smoothed version,
        and gently corrects toward it. This catches steps from ANY upstream
        processing (AudioSR, DSRE, etc.) — not just from this class.

        Only corrects above 4 kHz to preserve tonal character.
        """
        cfg = self.config
        strength = float(cfg.VITALITY_SMOOTH_STRENGTH)
        smooth_oct = float(cfg.VITALITY_SMOOTH_WIDTH_OCT)

        if strength < 0.01:
            return audio

        n_fft = cfg.HFR_FRAME_LENGTH
        hop = cfg.HFR_HOP_LENGTH
        eps = 1e-10
        result = audio.copy()

        num_bins = n_fft // 2 + 1
        bin_hz = (sr / 2.0) / num_bins

        # Sigma in bins (0.4 octaves at ~5kHz reference)
        ref_freq = 5000.0
        ref_oct_width_hz = ref_freq * (2.0 ** smooth_oct - 1.0)
        sigma_bins = max(12, int(ref_oct_width_hz / bin_hz))

        protect_below_hz = 4000.0
        protect_bin = int(protect_below_hz / bin_hz)

        for ch in range(audio.shape[1] if audio.ndim == 2 else 1):
            x = audio[:, ch] if audio.ndim == 2 else audio
            S = self.stft_engine.stft(x.astype(np.float64))
            mag = np.abs(S) + eps
            num_b, num_frames = mag.shape

            # Time-averaged spectral envelope in log domain
            avg_log = np.mean(np.log(mag), axis=1)

            # Smooth it
            try:
                avg_smooth = ndimage.gaussian_filter1d(
                    avg_log.astype(np.float32), sigma=float(sigma_bins),
                    mode="nearest"
                ).astype(np.float64)
            except Exception:
                continue

            # Correction
            correction = avg_smooth - avg_log

            # Protect below 4 kHz
            fade_width = max(1, protect_bin // 2)
            for b in range(num_b):
                if b < protect_bin:
                    correction[b] = 0.0
                elif b < protect_bin + fade_width:
                    t = float(b - protect_bin) / float(fade_width)
                    correction[b] *= (t * t)

            # Limit and scale
            correction = np.clip(correction, -0.25, 0.7)  # ~2dB cut, ~6dB boost max (raised from 0.5 to help fill HF)
            correction *= strength

            # Smooth the correction itself
            try:
                correction = ndimage.gaussian_filter1d(
                    correction.astype(np.float32), sigma=max(4, sigma_bins // 4),
                    mode="nearest"
                ).astype(np.float64)
            except Exception:
                pass

            # Apply as gain
            gain = np.exp(correction)
            S_out = S * gain[:, None]

            y = self.stft_engine.istft(S_out, length=len(x))
            if audio.ndim == 2:
                result[:, ch] = y
            else:
                result = y

        # Preserve peak
        orig_peak = np.max(np.abs(audio)) + eps
        res_peak = np.max(np.abs(result)) + eps
        if res_peak > orig_peak:
            result *= (orig_peak / res_peak)

        return result


# =========================
# HFP V2 — MDCT-based High-Frequency Prediction with Harmonic Structure Analysis
# =========================

class HFPv2Restorer:
    """HFP V2: MDCT-based High-Frequency Prediction.

    Uses Modified Discrete Cosine Transform with harmonic structure analysis
    for intelligent high-frequency restoration.  Algorithm (based on
    HRAudioWizard by Super-YH):

    1. Mid/Side decomposition for stereo processing
    2. Transient detection via onset strength
    3. MDCT transformation → HPSS separation (harmonic vs percussive)
    4. Cepstrum-based harmonic structure analysis → find fundamentals
    5. Overtone extrapolation above bandwidth cutoff via correlation
    6. Griffin-Lim phase reconstruction for synthesized content
    7. Spectral connection with smooth Linkwitz-Riley crossfade

    Key difference from V1/AnalogVitality (saturation-based):
    - Overtones are harmonically related to actual content, not generic tanh harmonics
    - HPSS treats tonal and transient content differently
    - Cepstrum finds real fundamentals for accurate harmonic series extension
    """

    N_FFT = 4096
    HOP = 1024
    MDCT_BLOCK = 2048       # MDCT block size (N/2 coefficients per frame)
    GRIFFIN_LIM_ITER = 32   # Phase reconstruction iterations
    HPSS_KERNEL = 31        # Median filter kernel for HPSS
    MAX_HARMONICS = 24      # Max overtones to measure for decay fitting
    MAX_EXTEND_HZ = 23000  # Don't extrapolate above this
    CEPSTRUM_LOW_HZ = 60    # Minimum f0 to detect
    CEPSTRUM_HIGH_HZ = 4000 # Maximum f0 to detect

    def __init__(self, config: UltimateGPUConfig):
        self.config = config
        self.n_fft = self.N_FFT
        self.hop = self.HOP
        self.window = np.hanning(self.n_fft).astype(np.float64)
        # MDCT window: sine window for perfect reconstruction
        N = self.MDCT_BLOCK * 2
        self.mdct_window = np.sin(np.pi / N * (np.arange(N) + 0.5)).astype(np.float64)

    # ─── public interface (same as AnalogVitality) ──────────────
    def restore(self, audio: np.ndarray, sr: int,
                lowpass_hz: Optional[int] = None) -> np.ndarray:
        """Apply HFPv2 restoration to audio."""
        original_peak = np.max(np.abs(audio)) + 1e-12
        cfg = self.config

        # Detect bandwidth
        if lowpass_hz is None or lowpass_hz <= 0:
            lowpass_hz = self._detect_bandwidth(audio, sr)

        nyq = sr / 2.0
        if lowpass_hz >= nyq * 0.92:
            print(f"      HFPv2: bandwidth {lowpass_hz:.0f} Hz ≈ Nyquist, skipping")
            return audio

        print(f"      HFPv2: detected bandwidth ~{lowpass_hz:.0f} Hz")
        print(f"      HFPv2: MDCT + cepstrum → harmonic extrapolation")

        amount = float(cfg.VITALITY_AMOUNT)
        dry = audio.copy()

        # Mid/Side decomposition for stereo
        if audio.ndim == 2 and audio.shape[1] == 2:
            mid  = (audio[:, 0] + audio[:, 1]) * 0.5
            side = (audio[:, 0] - audio[:, 1]) * 0.5
            mid_out  = self._process_channel(mid, sr, lowpass_hz, gain_scale=1.0)
            side_out = self._process_channel(side, sr, lowpass_hz, gain_scale=0.7)
            audio = np.column_stack([mid_out + side_out, mid_out - side_out])
        elif audio.ndim == 2:
            for ch in range(audio.shape[1]):
                audio[:, ch] = self._process_channel(audio[:, ch], sr, lowpass_hz)
        else:
            audio = self._process_channel(audio, sr, lowpass_hz)

        # Wet/dry
        audio = dry * (1.0 - amount) + audio * amount

        # Preserve peak
        out_peak = np.max(np.abs(audio)) + 1e-12
        if out_peak > original_peak:
            audio *= (original_peak / out_peak)

        return audio

    # ─── per-channel processing ─────────────────────────────────
    def _process_channel(self, x: np.ndarray, sr: int,
                         cutoff_hz: float, gain_scale: float = 1.0) -> np.ndarray:
        x = x.astype(np.float64)
        n = len(x)
        eps = 1e-12

        # 1. STFT for analysis
        S = self._stft(x)
        mag = np.abs(S) + eps
        phase = np.angle(S)
        num_bins, num_frames = S.shape

        cutoff_bin = max(1, int(cutoff_hz * self.n_fft / sr))
        if cutoff_bin >= num_bins - 4:
            return x

        # 2. HPSS — median filter separation
        H_mag, P_mag = self._hpss(mag)

        # 3. Onset detection for transient-aware processing
        onset_env = self._onset_strength(mag)

        # 4. Per-frame harmonic extrapolation
        extended = np.zeros_like(mag)

        # ── Energy envelope for dynamic modulation ──
        # Compute per-frame energy below cutoff; scale HF extension proportionally
        # so quiet frames produce proportionally quieter HF content.
        frame_energy = np.sum(mag[:cutoff_bin, :] ** 2, axis=0)  # (num_frames,)
        frame_energy_db = 10.0 * np.log10(frame_energy + eps)
        peak_energy_db = np.max(frame_energy_db)
        # Dynamic range: frames quieter than peak by >40 dB get near-zero HF
        energy_ratio = np.clip(
            10.0 ** ((frame_energy_db - peak_energy_db) / 20.0),
            0.0, 1.0
        )
        # Smooth slightly to avoid pumping, but keep attack fast
        try:
            energy_ratio = ndimage.uniform_filter1d(
                energy_ratio.astype(np.float32), size=3, mode='nearest'
            ).astype(np.float64)
        except Exception:
            pass

        for t in range(num_frames):
            frame_h = H_mag[:, t]
            frame_p = P_mag[:, t]
            is_transient = onset_env[t] > np.median(onset_env) * 2.0

            # Check if frame has enough energy (relative, not absolute)
            if energy_ratio[t] < 0.01:
                continue

            if is_transient:
                # Transients: noise-shaped extension (percussive content)
                extended[:, t] = self._extend_percussive(
                    frame_h + frame_p, cutoff_bin, num_bins, gain_scale)
            else:
                # Harmonic analysis via cepstrum
                f0_hz = self._cepstrum_f0(frame_h[:cutoff_bin], sr)

                if f0_hz is not None and f0_hz > self.CEPSTRUM_LOW_HZ:
                    # Extrapolate harmonic overtones
                    extended[:, t] = self._extrapolate_harmonics(
                        frame_h, f0_hz, sr, cutoff_bin, num_bins, gain_scale)
                else:
                    # No clear fundamental — use spectral envelope extension
                    extended[:, t] = self._extrapolate_envelope(
                        frame_h, cutoff_bin, num_bins, gain_scale)

                # Add subtle percussive extension too
                perc_ext = self._extend_percussive(
                    frame_p, cutoff_bin, num_bins, gain_scale * 0.3)
                extended[:, t] += perc_ext

            # ── Dynamic modulation: scale HF by frame energy ratio ──
            extended[cutoff_bin:, t] *= energy_ratio[t]

        # 5. Build new-HF-only spectrogram with crossfade region
        new_hf = np.zeros_like(extended)
        fade_bins = max(4, cutoff_bin // 10)
        fade_start = max(0, cutoff_bin - fade_bins // 2)

        for b in range(num_bins):
            if b < fade_start:
                new_hf[b, :] = 0.0          # Keep original below
            elif b < cutoff_bin:
                t_fade = (b - fade_start) / max(1, cutoff_bin - fade_start)
                new_hf[b, :] = extended[b, :] * (t_fade ** 2) * 0.2
            else:
                new_hf[b, :] = extended[b, :]

        # Transient-gated temporal smoothing: only smooth sustained frames,
        # preserve temporal sharpness at onsets/attacks.
        onset_peak = np.max(onset_env) + eps
        onset_norm = onset_env / onset_peak
        # Frames with onset_norm > 0.15 are transient-ish → no temporal blur
        sustain_weight = np.clip(1.0 - onset_norm / 0.15, 0.0, 1.0)
        # Expand transient protection: each onset protects ~3 frames after it
        try:
            sustain_weight = ndimage.minimum_filter1d(
                sustain_weight.astype(np.float32), size=5, origin=-2
            ).astype(np.float64)
        except Exception:
            pass

        try:
            for b in range(cutoff_bin, num_bins):
                row = new_hf[b, :]
                row_smooth = ndimage.gaussian_filter1d(
                    row.astype(np.float32), sigma=1.5, mode='nearest'
                ).astype(np.float64)
                # Blend: transient frames keep raw, sustained frames get smoothed
                new_hf[b, :] = row * (1.0 - sustain_weight) + row_smooth * sustain_weight
        except Exception:
            pass

        # 6. Griffin-Lim phase reconstruction for the synthesized HF
        hf_audio = self._griffin_lim(new_hf, n_iter=self.GRIFFIN_LIM_ITER,
                                      init_phase=phase)

        # 7. Trim / pad to match
        if len(hf_audio) > n:
            hf_audio = hf_audio[:n]
        elif len(hf_audio) < n:
            hf_audio = np.pad(hf_audio, (0, n - len(hf_audio)))

        # 8. 23 kHz lowpass on synthesized HF (from Jarredou fork — removes
        #    ultrasonic artifacts above audible range)
        if sr >= 46000:
            try:
                sos = signal.butter(2, 23000, btype='lowpass', fs=sr, output='sos')
                hf_audio = signal.sosfiltfilt(sos, hf_audio).astype(np.float64)
            except Exception:
                pass

        # 9. Add synthesized HF to original
        output = x + hf_audio
        return output

    # ─── STFT / ISTFT ───────────────────────────────────────────
    def _stft(self, x: np.ndarray) -> np.ndarray:
        """Forward STFT with Hann window (uses librosa for speed)."""
        if LIBROSA_AVAILABLE:
            try:
                return librosa.stft(x.astype(np.float32), n_fft=self.n_fft,
                                    hop_length=self.hop, window='hann')
            except (AttributeError, TypeError):
                pass  # Fall through to manual loop
        # Fallback: manual loop
        padded = np.pad(x, (self.n_fft // 2, self.n_fft // 2), mode='reflect')
        n_frames = 1 + (len(padded) - self.n_fft) // self.hop
        S = np.zeros((self.n_fft // 2 + 1, n_frames), dtype=np.complex128)
        for i in range(n_frames):
            start = i * self.hop
            frame = padded[start:start + self.n_fft] * self.window
            S[:, i] = np.fft.rfft(frame)
        return S

    def _istft(self, S: np.ndarray, length: int = 0) -> np.ndarray:
        """Inverse STFT with overlap-add (uses librosa for speed)."""
        if LIBROSA_AVAILABLE:
            try:
                return librosa.istft(S, hop_length=self.hop, window='hann',
                                     length=length if length > 0 else None)
            except (AttributeError, TypeError):
                pass  # Fall through to manual loop
        # Fallback: manual loop
        num_bins, n_frames = S.shape
        out_len = (n_frames - 1) * self.hop + self.n_fft
        y = np.zeros(out_len, dtype=np.float64)
        w_sum = np.zeros(out_len, dtype=np.float64)
        for i in range(n_frames):
            start = i * self.hop
            frame = np.fft.irfft(S[:, i], n=self.n_fft) * self.window
            y[start:start + self.n_fft] += frame
            w_sum[start:start + self.n_fft] += self.window ** 2
        w_sum = np.maximum(w_sum, 1e-8)
        y /= w_sum
        offset = self.n_fft // 2
        if length <= 0:
            length = max(1, out_len - self.n_fft)
        y = y[offset:offset + length]
        return y

    # ─── HPSS (Harmonic-Percussive Source Separation) ───────────
    def _hpss(self, mag: np.ndarray) -> tuple:
        """Separate harmonic and percussive via median filtering.

        Harmonic = horizontal median (across time → steady tones)
        Percussive = vertical median (across frequency → transients)
        """
        kernel = self.HPSS_KERNEL
        eps = 1e-10

        # Median filter in time direction → harmonic
        H_med = ndimage.median_filter(mag, size=(1, kernel))
        # Median filter in frequency direction → percussive
        P_med = ndimage.median_filter(mag, size=(kernel, 1))

        # Wiener-like soft masks
        H_mask = (H_med ** 2) / (H_med ** 2 + P_med ** 2 + eps)
        P_mask = (P_med ** 2) / (H_med ** 2 + P_med ** 2 + eps)

        return mag * H_mask, mag * P_mask

    # ─── onset strength for transient detection ─────────────────
    def _onset_strength(self, mag: np.ndarray) -> np.ndarray:
        """Compute onset strength envelope from spectral flux."""
        # Spectral flux = sum of positive differences across time
        diff = np.diff(mag, axis=1)
        diff = np.maximum(diff, 0)
        onset = np.sum(diff, axis=0)
        # Prepend zero for first frame
        onset = np.concatenate([[0.0], onset])
        return onset

    # ─── Cepstrum-based fundamental frequency detection ─────────
    def _cepstrum_f0(self, frame_mag: np.ndarray, sr: int) -> Optional[float]:
        """Find fundamental frequency via autocorrelation of log spectrum.

        Computes the autocorrelation of the log magnitude spectrum and finds
        the FIRST significant peak (not the strongest), which corresponds to
        the fundamental harmonic spacing.
        """
        eps = 1e-10
        n = len(frame_mag)
        if n < 32:
            return None

        bin_hz = sr / float(self.n_fft)

        # Log magnitude spectrum (centered)
        log_mag = np.log(frame_mag + eps)
        log_mag = log_mag - np.mean(log_mag)

        # Autocorrelation via FFT
        F = np.fft.rfft(log_mag, n=n * 2)
        acf = np.fft.irfft(F * np.conj(F))[:n]

        # Normalize
        if acf[0] > eps:
            acf = acf / acf[0]

        # Search range: lag in bins → f0 range
        min_lag = max(3, int(self.CEPSTRUM_LOW_HZ / bin_hz))
        max_lag = min(n // 2, int(self.CEPSTRUM_HIGH_HZ / bin_hz))

        if min_lag >= max_lag - 1:
            return None

        # Find the FIRST significant peak (not strongest) — avoids octave errors
        threshold = 0.12
        peak_idx = None
        for lag in range(min_lag, max_lag):
            if lag <= 0 or lag >= n - 1:
                continue
            # Local peak: higher than both neighbors and above threshold
            if (acf[lag] > threshold and
                acf[lag] >= acf[lag - 1] and
                acf[lag] >= acf[lag + 1]):
                peak_idx = lag
                break

        if peak_idx is None:
            return None

        # Parabolic interpolation for sub-bin accuracy
        if 0 < peak_idx < n - 1:
            alpha = acf[peak_idx - 1]
            beta = acf[peak_idx]
            gamma = acf[peak_idx + 1]
            denom = alpha - 2.0 * beta + gamma
            if abs(denom) > eps:
                delta = 0.5 * (alpha - gamma) / denom
                peak_idx_refined = peak_idx + delta
            else:
                peak_idx_refined = float(peak_idx)
        else:
            peak_idx_refined = float(peak_idx)

        # Convert lag (in spectral bins) to Hz
        f0_hz = peak_idx_refined * bin_hz

        if f0_hz < self.CEPSTRUM_LOW_HZ or f0_hz > self.CEPSTRUM_HIGH_HZ:
            return None

        return f0_hz

    # ─── Harmonic overtone extrapolation ────────────────────────
    def _extrapolate_harmonics(self, frame_mag: np.ndarray, f0_hz: float,
                                sr: int, cutoff_bin: int, num_bins: int,
                                gain_scale: float) -> np.ndarray:
        """Extend the harmonic series above cutoff using measured decay pattern.

        1. Identify existing harmonic positions (n * f0) below cutoff
        2. Measure their amplitudes → fit a log-linear decay curve
        3. Predict amplitudes of harmonics above the cutoff
        4. Place them at the correct frequency bins with natural width
        """
        result = frame_mag.copy()
        bin_hz = sr / float(self.n_fft)
        eps = 1e-12
        max_extend_bin = int(self.MAX_EXTEND_HZ / bin_hz)

        # Find existing harmonics below cutoff (up to MAX_HARMONICS for fitting)
        harmonic_amps = []
        harmonic_indices = []

        h = 1
        while True:
            freq = h * f0_hz
            b = int(freq / bin_hz + 0.5)
            if b >= cutoff_bin or b >= num_bins:
                break
            if b > 0:
                # Peak search: look ±2 bins around expected position
                lo = max(0, b - 2)
                hi = min(cutoff_bin, b + 3)
                local_peak = float(np.max(frame_mag[lo:hi]))
                if local_peak > eps:
                    harmonic_amps.append(local_peak)
                    harmonic_indices.append(h)
            h += 1
            if len(harmonic_amps) >= self.MAX_HARMONICS:
                break

        if len(harmonic_amps) < 2:
            return self._extrapolate_envelope(frame_mag, cutoff_bin, num_bins, gain_scale)

        # Fit log-linear decay: log(amp) = intercept + slope * harmonic_number
        h_arr = np.array(harmonic_indices, dtype=np.float64)
        a_arr = np.log(np.array(harmonic_amps, dtype=np.float64) + eps)

        n_h = len(h_arr)
        mean_h = np.mean(h_arr)
        mean_a = np.mean(a_arr)
        cov = np.sum((h_arr - mean_h) * (a_arr - mean_a))
        var = np.sum((h_arr - mean_h) ** 2) + eps
        slope = cov / var
        intercept = mean_a - slope * mean_h

        # Slope should be negative (harmonics decay) — clamp
        slope = min(slope, -0.01)

        # Reference level: maximum existing content for clamping
        max_existing = float(np.max(frame_mag[:cutoff_bin])) * 0.5

        # Extrapolate above cutoff until we exceed Nyquist or max extend
        h = 1
        while True:
            freq = h * f0_hz
            b = int(freq / bin_hz + 0.5)
            if b >= min(num_bins, max_extend_bin):
                break
            if b < cutoff_bin:
                h += 1
                continue  # Already exists

            predicted_log_amp = intercept + slope * h
            predicted_amp = np.exp(predicted_log_amp) * gain_scale
            predicted_amp = min(predicted_amp, max_existing)

            # Place with ±1 bin spread for natural width
            for db in range(-1, 2):
                bb = b + db
                if 0 <= bb < num_bins:
                    spread = 1.0 if db == 0 else 0.3
                    result[bb] = max(result[bb], predicted_amp * spread)
            h += 1

        return result

    # ─── Spectral envelope extrapolation (no clear f0) ──────────
    def _extrapolate_envelope(self, frame_mag: np.ndarray,
                               cutoff_bin: int, num_bins: int,
                               gain_scale: float) -> np.ndarray:
        """Extend spectral envelope above cutoff by extrapolating the decay trend.

        Used when no clear fundamental frequency is detected (noise, complex textures).
        Fits the spectral slope near the cutoff and extends it with added noise modulation.
        """
        result = frame_mag.copy()
        eps = 1e-12

        # Measure spectral slope in the octave below cutoff
        analysis_start = max(1, cutoff_bin // 2)
        region = frame_mag[analysis_start:cutoff_bin]
        if len(region) < 8:
            return result

        # Log-linear fit of the region
        bins_r = np.arange(len(region), dtype=np.float64) + analysis_start
        log_r = np.log(region + eps)

        # Simple slope
        n_r = len(bins_r)
        mean_b = np.mean(bins_r)
        mean_l = np.mean(log_r)
        cov = np.sum((bins_r - mean_b) * (log_r - mean_l))
        var = np.sum((bins_r - mean_b) ** 2) + eps
        slope = cov / var
        intercept = mean_l - slope * mean_b

        # Steepen the slope slightly for natural rolloff
        slope = min(slope, -0.001) * 1.2

        # Extrapolate with gentle noise modulation
        np.random.seed(cutoff_bin)  # Reproducible per frame
        for b in range(cutoff_bin, num_bins):
            predicted = np.exp(intercept + slope * b) * gain_scale
            # Add ±20% random variation for natural texture
            noise_mod = 1.0 + 0.2 * (np.random.random() - 0.5)
            result[b] = max(result[b], predicted * noise_mod)

        return result

    # ─── Percussive extension (noise-shaped) ────────────────────
    def _extend_percussive(self, frame_mag: np.ndarray,
                            cutoff_bin: int, num_bins: int,
                            gain_scale: float) -> np.ndarray:
        """Extend percussive content above cutoff using noise-shaped energy.

        Transients naturally have broad spectral content. We extend
        the spectral envelope with shaped noise that follows the
        energy decay pattern of the existing percussive content.
        """
        result = np.zeros(num_bins, dtype=np.float64)
        eps = 1e-12

        # Reference energy near cutoff (average of last octave)
        ref_start = max(1, cutoff_bin * 3 // 4)
        ref_energy = np.mean(frame_mag[ref_start:cutoff_bin]) + eps

        # Decay rate: measure slope in existing region
        if cutoff_bin > 16:
            low_energy = np.mean(frame_mag[cutoff_bin // 4:cutoff_bin // 2]) + eps
            high_energy = np.mean(frame_mag[cutoff_bin // 2:cutoff_bin]) + eps
            # dB per octave
            db_per_oct = 20.0 * np.log10(high_energy / low_energy)
            db_per_oct = np.clip(db_per_oct, -12.0, -1.0)
        else:
            db_per_oct = -6.0

        for b in range(cutoff_bin, num_bins):
            # Distance in octaves from cutoff
            octaves = np.log2(max(1, b) / max(1, cutoff_bin))
            energy = ref_energy * (10.0 ** (db_per_oct * octaves / 20.0))
            result[b] = energy * gain_scale

        return result

    # ─── Griffin-Lim phase reconstruction ───────────────────────
    def _griffin_lim(self, target_mag: np.ndarray, n_iter: int = 32,
                      init_phase: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct phase for a magnitude spectrogram via Griffin-Lim iteration.

        If init_phase is provided, uses it as starting point (warm start).
        """
        num_bins, n_frames = target_mag.shape
        # Expected output length (auto-computed from spectrogram dimensions)
        expected_len = (n_frames - 1) * self.hop

        if init_phase is not None:
            phase = init_phase.copy()
        else:
            phase = np.random.uniform(-np.pi, np.pi, target_mag.shape)

        for iteration in range(n_iter):
            S = target_mag * np.exp(1j * phase)
            y = self._istft(S)  # Auto length
            if len(y) == 0:
                break
            S_new = self._stft(y)

            # Match shapes (STFT may produce slightly different frame count)
            min_frames = min(n_frames, S_new.shape[1])
            new_phase = np.angle(S_new[:, :min_frames])

            # Only update phase where we have target energy
            mask = target_mag[:, :min_frames] > 1e-10
            phase[:, :min_frames] = np.where(mask, new_phase, phase[:, :min_frames])

        # Final synthesis
        S_final = target_mag * np.exp(1j * phase)
        return self._istft(S_final, length=expected_len)

    # ─── bandwidth detection (shared with AnalogVitality) ───────
    def _detect_bandwidth(self, audio: np.ndarray, sr: int) -> int:
        """Detect effective bandwidth from spectral energy drop."""
        if audio.ndim == 2:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio
        mono = mono.astype(np.float64)
        eps = 1e-12

        S = self._stft(mono)
        mag = np.abs(S)
        e = np.mean(mag, axis=1) + eps

        try:
            e_s = ndimage.gaussian_filter1d(e.astype(np.float32), sigma=2.0,
                                             mode="nearest").astype(np.float64)
        except Exception:
            e_s = e

        e_db = 20.0 * np.log10(e_s + eps)
        peak_db = float(np.max(e_db))

        bin_hz = sr / float(self.n_fft)
        bw_hint = getattr(self.config, '_detected_bandwidth', 15000.0)
        min_hz = max(2000.0, bw_hint * 0.50)
        max_hz = max(min_hz + 1000.0, (sr / 2.0) - 1500.0)
        min_bin = max(16, int(min_hz / bin_hz))
        max_bin = min(len(e_db) - 16, int(max_hz / bin_hz))

        drop_db = 34.0
        cutoff = None
        for b in range(min_bin, max_bin):
            tail = e_db[b:min(len(e_db), b + 48)]
            if tail.size < 24:
                break
            if np.median(tail) < (peak_db - drop_db):
                cutoff = b
                break

        if cutoff is None:
            de = np.diff(e_db)
            region = de[min_bin:max_bin]
            cutoff = (int(np.argmin(region) + min_bin) if region.size > 0
                      else int(self.n_fft // 4))

        cutoff = int(np.clip(cutoff, 48, len(e_db) - 8))
        return int(cutoff * bin_hz)


def _transient_peak_match_upward_only(out_audio: np.ndarray,
                                      ref_audio: np.ndarray,
                                      sr: int,
                                      strength: float = 0.65,
                                      max_boost_db: float = 6.0) -> np.ndarray:
    """Restore the first milliseconds of attacks by UPWARD-ONLY peak-envelope matching.

    This avoids copying the reference waveform (less timbral bleed), but brings back
    the *amplitude* of the first milliseconds when they got rounded by SR/declipping.

    Mechanism:
      - estimate a fast peak envelope for ref and out (≈1 ms)
      - build a transient mask from the reference (peak/rms)
      - compute required gain = ref_env / out_env (clamped, upward-only)
      - smooth gain (very fast attack, short release)
      - apply gain under the transient mask only

    If out is already sharper than ref, we do nothing.
    """
    if ref_audio is None or out_audio is None:
        return out_audio
    # Align lengths when they differ slightly (e.g. after resampling)
    if out_audio.ndim != ref_audio.ndim:
        return out_audio
    if out_audio.ndim >= 2 and out_audio.shape[1] != ref_audio.shape[1]:
        return out_audio
    min_len = min(out_audio.shape[0], ref_audio.shape[0])
    if min_len < 64:
        return out_audio
    # Work on aligned slice; result will be padded back at the end
    _orig_out_len = out_audio.shape[0]
    _aligned_out = out_audio[:min_len]
    _aligned_ref = ref_audio[:min_len]

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-6:
        return out_audio

    eps = 1e-12
    y = _aligned_out.astype(np.float64, copy=False)
    r = _aligned_ref.astype(np.float64, copy=False)

    # mono for detection
    r_m = np.mean(r, axis=1)
    y_m = np.mean(y, axis=1)

    # fast peak envelope (~1 ms)
    peak_win = max(8, int(sr * 0.001))
    r_env = ndimage.maximum_filter1d(np.abs(r_m), size=peak_win, mode="nearest") + eps
    y_env = ndimage.maximum_filter1d(np.abs(y_m), size=peak_win, mode="nearest") + eps

    # rms envelope (~10 ms) for transientness
    rms_win = max(32, int(sr * 0.010))
    k = np.ones(rms_win, dtype=np.float64) / float(rms_win)
    r_rms = np.sqrt(np.convolve(r_m * r_m, k, mode="same") + eps)

    transientness = r_env / (r_rms + eps)
    t0, t1 = 1.35, 2.40
    mask = np.clip((transientness - t0) / (t1 - t0 + eps), 0.0, 1.0)

    # smooth the mask (attack ~0.5ms, release ~6ms)
    a = max(1, int(sr * 0.0005))
    b = max(1, int(sr * 0.006))
    mask = ndimage.uniform_filter1d(mask, size=a, mode="nearest")
    mask = ndimage.uniform_filter1d(mask, size=b, mode="nearest")
    mask = np.clip(mask, 0.0, 1.0)

    # required gain to match ref peak env (upward only)
    g = (r_env / y_env)
    g = np.maximum(1.0, g)

    max_boost = 10.0 ** (float(max_boost_db) / 20.0)
    g = np.minimum(g, max_boost)

    # smooth gain (attack ~0.25ms, release ~10ms)
    ga = max(1, int(sr * 0.00025))
    gr = max(1, int(sr * 0.010))
    g = ndimage.uniform_filter1d(g, size=ga, mode="nearest")
    g = ndimage.uniform_filter1d(g, size=gr, mode="nearest")

    # apply under mask, strength-scaled
    g_eff = 1.0 + (g - 1.0) * (mask * strength)

    out = y.copy()
    out[:, 0] *= g_eff
    if out.shape[1] > 1:
        out[:, 1] *= g_eff

    # Restore original length (pad with unmodified tail if out_audio was longer)
    result = out.astype(out_audio.dtype, copy=False)
    if _orig_out_len > min_len:
        result = np.concatenate([result, out_audio[min_len:]], axis=0)
    return result


# =========================
# Clipping Detection
# =========================

class ClippingDetector:
    """Clipping detection"""
    
    @staticmethod
    def detect_clipping(audio: np.ndarray, theta: float = 0.95) -> Tuple[bool, float, dict]:
        audio_flat = audio.flatten()
        audio_abs = np.abs(audio_flat)
        max_val = np.max(audio_abs)
        
        if max_val == 0:
            return False, 0.0, {}
        
        threshold = theta
        clipped_samples = np.sum(np.abs(audio_flat) > threshold)
        clipping_ratio = clipped_samples / len(audio_flat)
        
        hist, bin_edges = np.histogram(audio_abs, bins=500)
        top_bins = hist[-10:]
        hist_spike = np.max(top_bins) > np.median(hist) * 20
        
        # Estimate thresholds
        positive_threshold = ClippingDetector._estimate_threshold(audio_flat[audio_flat > 0])
        negative_threshold = ClippingDetector._estimate_threshold(-audio_flat[audio_flat < 0])
        
        details = {
            'clipping_ratio': clipping_ratio,
            'positive_threshold': positive_threshold,
            'negative_threshold': negative_threshold,
            'max_amplitude': max_val,
            'num_clipped_samples': int(clipped_samples),
        }
        
        is_clipped = clipping_ratio > 0.005 or hist_spike
        return is_clipped, clipping_ratio, details
    
    @staticmethod
    def _estimate_threshold(audio_positive: np.ndarray) -> float:
        if len(audio_positive) == 0:
            return 1.0
        
        hist, bin_edges = np.histogram(audio_positive, bins=500)
        hist_smooth = ndimage.gaussian_filter1d(hist.astype(float), sigma=2)
        
        upper_region = int(len(hist) * 0.8)
        upper_hist = hist_smooth[upper_region:]
        
        if len(upper_hist) > 0 and np.max(upper_hist) > np.mean(hist_smooth) * 5:
            peak_idx = np.argmax(upper_hist) + upper_region
            return bin_edges[peak_idx]
        
        return np.percentile(audio_positive, 99.5)


# =========================
# Resampling
# =========================

def resample_gpu(audio: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    """GPU-accelerated resampling using PyTorch"""
    if in_sr == out_sr:
        return audio
    
    if BACKEND == "torch" and torch is not None and hasattr(torchaudio.transforms, 'Resample'):
        is_stereo = audio.ndim == 2
        
        if is_stereo:
            audio_gpu = torch.from_numpy(np.ascontiguousarray(audio.T)).to(torch.float32).cuda()
        else:
            audio_gpu = torch.from_numpy(np.ascontiguousarray(audio)).to(torch.float32).unsqueeze(0).cuda()
        
        resampler = torchaudio.transforms.Resample(
            orig_freq=in_sr,
            new_freq=out_sr,
            lowpass_filter_width=192,
            rolloff=0.9975,
            resampling_method='sinc_interp_kaiser',
        ).cuda()
        
        resampled = resampler(audio_gpu)
        
        if is_stereo:
            return resampled.T.cpu().numpy().astype(np.float64)
        else:
            return resampled.squeeze(0).cpu().numpy().astype(np.float64)
    else:
        # CPU fallback using scipy
        from scipy.signal import resample_poly
        gcd = math.gcd(in_sr, out_sr)
        up = out_sr // gcd
        down = in_sr // gcd
        return resample_poly(audio, up, down, axis=0)


# =========================
# Soft Limiter
# =========================

def soft_limiter(audio: np.ndarray, threshold: float = 0.90, 
                 ratio: float = 20.0, knee: float = 0.05) -> np.ndarray:
    """Improved soft limiter with soft knee"""
    audio = np.ascontiguousarray(audio)
    audio_abs = np.abs(audio)
    
    # Soft knee compression
    gain = np.ones_like(audio_abs)
    
    # Below knee start - unity gain
    knee_start = threshold - knee
    knee_end = threshold + knee
    
    # In knee region - gradual compression
    in_knee = (audio_abs >= knee_start) & (audio_abs <= knee_end)
    if np.any(in_knee):
        knee_input = audio_abs[in_knee]
        # Quadratic knee curve
        knee_factor = (knee_input - knee_start) / (2 * knee)
        knee_gain = 1 - (1 - 1/ratio) * knee_factor ** 2
        gain[in_knee] = knee_gain
    
    # Above knee - full compression
    above_knee = audio_abs > knee_end
    if np.any(above_knee):
        above_input = audio_abs[above_knee]
        compressed = knee_end + (above_input - knee_end) / ratio
        gain[above_knee] = compressed / above_input
    
    # Smooth the gain to avoid artifacts
    if audio.ndim == 1:
        gain = ndimage.uniform_filter1d(gain, size=64)
    else:
        gain = ndimage.uniform_filter1d(gain, size=64, axis=0)
    
    return audio * gain


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak LIMIT (downward only).

    Important: this intentionally never *raises* gain, because that turns any
    peak reduction/limiting earlier in the chain into audible upward
        compression (quiet parts get louder).
    """
    target_peak = float(target_peak)
    if target_peak <= 0:
        return audio
    current_peak = float(np.max(np.abs(audio)) + 1e-12)
    if current_peak > target_peak:
        return audio * (target_peak / current_peak)
    return audio



def _preserve_envelope_down_only(out_audio: np.ndarray,
                                 ref_audio: np.ndarray,
                                 sr: int,
                                 win_ms: float = 80.0,
                                 hop_ms: float = 20.0,
                                 silence_db: float = -65.0,
                                 max_atten_db: float = 24.0) -> np.ndarray:
    """Attenuate output so its short-term RMS envelope never exceeds the reference.

    This is a "downward-only" dynamics match:
      - If a stage (AudioSR/Demucs/etc) internally pushes fades/quiet parts up,
        we push them back down to the original envelope.
      - If output is quieter than reference, we do NOT boost (prevents noise lift).

    Stereo: gain is derived from MID (L+R) and applied equally to both channels
    to preserve stereo balance/width.
    """
    try:
        y = np.asarray(out_audio, dtype=np.float64)
        x = np.asarray(ref_audio, dtype=np.float64)
    except Exception:
        return out_audio

    # Ensure 2D (N, C)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim == 1:
        x = x[:, None]

    # Match channels (best effort)
    if x.shape[1] != y.shape[1]:
        if x.shape[1] == 1 and y.shape[1] == 2:
            x = np.repeat(x, 2, axis=1)
        elif x.shape[1] == 2 and y.shape[1] == 1:
            x = np.mean(x, axis=1, keepdims=True)
        else:
            x = x[:, :y.shape[1]] if x.shape[1] > y.shape[1] else np.pad(x, ((0, 0), (0, y.shape[1] - x.shape[1])))

    # Match length (start-aligned truncation/padding — center-crop would
    # misalign the reference temporally and produce wrong gain envelopes)
    n = y.shape[0]
    if x.shape[0] != n:
        if x.shape[0] > n:
            x = x[:n, :]
        else:
            pad = n - x.shape[0]
            x = np.pad(x, ((0, pad), (0, 0)), mode='edge')

    sr = int(sr)
    win = max(64, int(sr * (win_ms / 1000.0)))
    hop = max(16, int(sr * (hop_ms / 1000.0)))

    # Mid for gain detection (preserves stereo image)
    x_mid = np.mean(x, axis=1)
    y_mid = np.mean(y, axis=1)

    # RMS envelope via moving average of squared signal
    eps = 1e-12
    try:
        x_env2 = ndimage.uniform_filter1d(x_mid * x_mid, size=win, mode="nearest")
        y_env2 = ndimage.uniform_filter1d(y_mid * y_mid, size=win, mode="nearest")
    except Exception:
        # fallback: simple convolution (slower)
        w = np.ones(win, dtype=np.float64) / float(win)
        x_env2 = np.convolve(x_mid * x_mid, w, mode="same")
        y_env2 = np.convolve(y_mid * y_mid, w, mode="same")

    x_env = np.sqrt(x_env2 + eps)
    y_env = np.sqrt(y_env2 + eps)

    # Sampled envelope points
    idx = np.arange(0, n, hop, dtype=np.int64)
    x_p = x_env[idx]
    y_p = y_env[idx]

    silence_floor = 10.0 ** (float(silence_db) / 20.0)
    ref = np.maximum(x_p, silence_floor)

    g = ref / (y_p + eps)          # <1 when output louder than ref
    g = np.minimum(1.0, g)         # never boost
    min_g = 10.0 ** (-float(max_atten_db) / 20.0)
    g = np.clip(g, min_g, 1.0)

    # Smooth gain (avoid pumping)
    try:
        g = ndimage.gaussian_filter1d(g.astype(np.float32), sigma=1.2, mode="nearest").astype(np.float64)
        g = np.clip(g, min_g, 1.0)
    except Exception:
        pass

    # Upsample gain to per-sample by linear interpolation
    g_full = np.interp(np.arange(n, dtype=np.float64),
                       idx.astype(np.float64),
                       g.astype(np.float64)).astype(np.float64)

    y2 = y * g_full[:, None]

    # Return with original dtype/layout
    if out_audio.ndim == 1:
        return y2[:, 0].astype(out_audio.dtype, copy=False)
    return y2.astype(out_audio.dtype, copy=False)


# =========================
# Parameter Tuning
# =========================

class ParameterTuner:
    """Audio analysis and parameter tuning"""
    
    @staticmethod
    def analyze(audio: np.ndarray, sr: int) -> dict:
        """Analyze audio characteristics with robust bandwidth detection"""
        audio_mono = audio.flatten() if audio.ndim == 2 else audio
        
        results = {
            'sample_rate': sr,
            'duration': len(audio_mono) / sr,
            'channels': audio.shape[1] if audio.ndim == 2 else 1,
        }
        
        # Spectral analysis
        nperseg = min(4096, len(audio_mono) // 4)
        f, psd = signal.welch(audio_mono, sr, nperseg=nperseg)
        
        total_energy = np.sum(psd)
        hf_mask = f >= 8000
        results['hf_energy_ratio'] = np.sum(psd[hf_mask]) / (total_energy + 1e-12)
        results['spectral_centroid'] = np.sum(f * psd) / (total_energy + 1e-12)
        
        # Dynamics
        results['peak'] = np.max(np.abs(audio_mono))
        results['rms'] = np.sqrt(np.mean(audio_mono ** 2))
        
        # ─── Robust bandwidth detection ─────────────────────────────
        # Smooth the PSD and convert to dB
        psd_smooth = ndimage.gaussian_filter1d(psd.astype(np.float32), sigma=5.0).astype(np.float64)
        psd_db = 10.0 * np.log10(psd_smooth + 1e-20)
        
        # Find the peak level in the 500–5000 Hz band (where most content lives)
        core_mask = (f >= 500) & (f <= 5000)
        if np.any(core_mask):
            core_peak_db = np.max(psd_db[core_mask])
        else:
            core_peak_db = np.max(psd_db)
        
        # Bandwidth = highest frequency where PSD is within 30 dB of core peak
        # This catches both gradual rolloffs and sharp cutoffs
        above_floor = psd_db >= (core_peak_db - 30.0)
        above_indices = np.where(above_floor)[0]
        if len(above_indices) > 0:
            bw_hz = float(f[above_indices[-1]])
        else:
            bw_hz = float(sr / 2.0)
        
        # Sanity: clamp to [1000, Nyquist - 1000]
        nyquist = sr / 2.0
        bw_hz = float(np.clip(bw_hz, 1000.0, nyquist - 1000.0))
        
        # Secondary check: gradient-based cliff detection.
        # Walk DOWN from the spectral-cliff estimate and look for the steepest
        # spectral drop.  If we find a sharp cliff well below our estimate it
        # means there was broadband noise/aliasing extending ABOVE the true
        # content edge.  Only nudge the estimate down if the cliff is truly
        # dramatic (> 12 dB in a narrow window) AND the cliff is at least 60%
        # of bw_hz (never halve the estimate just because of spectral slope).
        df = f[1] - f[0] if len(f) > 1 else 1.0
        grad_win_hz = 500.0                      # look for drops within 500 Hz
        grad_win_bins = max(2, int(grad_win_hz / (df + 1e-6)))
        bw_bin = np.searchsorted(f, bw_hz)
        search_lo_bin = np.searchsorted(f, bw_hz * 0.60)
        if bw_bin > search_lo_bin + grad_win_bins:
            grad = np.zeros(bw_bin - search_lo_bin)
            for gi in range(len(grad)):
                b = search_lo_bin + gi
                b_end = min(len(psd_db), b + grad_win_bins)
                grad[gi] = psd_db[b] - np.mean(psd_db[b:b_end])
            cliff_idx = int(np.argmax(grad))
            cliff_db = float(grad[cliff_idx])
            if cliff_db > 12.0:
                cliff_hz = float(f[search_lo_bin + cliff_idx])
                # Only lower the estimate, never raise; and never below 60% of
                # the spectral-cliff estimate.
                bw_hz = max(bw_hz * 0.60, min(bw_hz, cliff_hz))
        
        bw_hz = float(np.clip(bw_hz, 1000.0, nyquist - 1000.0))
        
        results['effective_bandwidth'] = bw_hz
        
        return results
    
    @staticmethod
    def suggest_parameters(analysis: dict, config: UltimateGPUConfig) -> UltimateGPUConfig:
        """Adapt all frequency-dependent parameters based on detected bandwidth"""
        bw = analysis['effective_bandwidth']
        sr = analysis['sample_rate']
        nyquist = sr / 2.0
        
        if analysis['hf_energy_ratio'] < 0.03:
            config.DSRE_DECAY = max(0.8, config.DSRE_DECAY - 0.15)
            config.VITALITY_SHELF_GAIN_DB = min(7.0, config.VITALITY_SHELF_GAIN_DB + 2.0)
        
        if bw < 8000:
            config.DSRE_M = min(16, config.DSRE_M + 4)
            config.VITALITY_SATURATION_DRIVE = min(0.3, config.VITALITY_SATURATION_DRIVE + 0.05)
        
        # ─── Bandwidth-adaptive frequency parameters ────────────────
        # Store detected bandwidth on the config so downstream stages can use it
        config._detected_bandwidth = bw
        
        # DSRE: source content above 30% of bandwidth, keep output above 90%
        config.DSRE_PRE_HP = max(1000.0, bw * 0.30)
        config.DSRE_POST_HP = max(2000.0, bw * 0.90)
        
        # Naturalize: start blurring at the bandwidth boundary
        config.NATURALIZE_TRANSITION_HZ = max(2000.0, bw * 0.95)
        
        # Pre-AudioSR lowpass: clean cutoff just above bandwidth
        # (redundant when feed_sr < orig_sr since downsampling creates clean Nyquist,
        #  but helps when original SR is already at/below 2× BW)
        config.PRE_LPF_HZ = max(2000.0, bw * 1.02)
        
        # Dynamic seam bridge: derive search/reference ranges from bandwidth
        # Search window: 70% to 130% of bandwidth
        config.DYNAMIC_SEAM_SEARCH_LO = max(1500.0, bw * 0.70)
        config.DYNAMIC_SEAM_SEARCH_HI = min(nyquist - 500, bw * 1.30)
        # Reference energy band: 75% to 95% of bandwidth (just below the seam)
        config.DYNAMIC_SEAM_REF_LO = max(1000.0, bw * 0.75)
        config.DYNAMIC_SEAM_REF_HI = max(1500.0, bw * 0.95)
        # Offset: proportional to bandwidth (~4% of BW)
        config.DYNAMIC_SEAM_OFFSET_HZ = max(100.0, bw * 0.04)
        
        print(f"  ─ Bandwidth-adaptive tuning for {bw:.0f} Hz:")
        print(f"    DSRE: pre_hp={config.DSRE_PRE_HP:.0f}, post_hp={config.DSRE_POST_HP:.0f}")
        print(f"    Naturalize transition: {config.NATURALIZE_TRANSITION_HZ:.0f} Hz")
        print(f"    Pre-LPF: {config.PRE_LPF_HZ:.0f} Hz")
        print(f"    Seam bridge: search [{config.DYNAMIC_SEAM_SEARCH_LO:.0f}–{config.DYNAMIC_SEAM_SEARCH_HI:.0f}], "
              f"ref [{config.DYNAMIC_SEAM_REF_LO:.0f}–{config.DYNAMIC_SEAM_REF_HI:.0f}], "
              f"offset {config.DYNAMIC_SEAM_OFFSET_HZ:.0f} Hz")
        
        return config



# =========================
# Black-box AI Music Fixer (ML + DSP)
# =========================


def _transient_reinject_from_ref(out_audio: np.ndarray, ref_audio: np.ndarray, sr: int,
                                 strength: float = 0.65) -> np.ndarray:
    """
    Restore the *first milliseconds* of attacks by blending back the reference only where
    it is clearly transient-dominant.

    Why: any soft-limiting / declipping / SR model can reduce crest factor (peak-to-RMS),
    making attacks feel less sharp even if sustain is gain-matched.

    This is intentionally conservative:
      - builds a transient mask from the reference (mono)
      - smoothly attacks/releases the mask
      - blends reference into the processed signal only under the mask
    """
    if ref_audio is None or out_audio is None:
        return out_audio
    # Align lengths when they differ slightly (e.g. after resampling)
    if out_audio.ndim != ref_audio.ndim:
        return out_audio
    if out_audio.ndim >= 2 and out_audio.shape[1] != ref_audio.shape[1]:
        return out_audio
    min_len = min(out_audio.shape[0], ref_audio.shape[0])
    if min_len < 64:
        return out_audio
    _orig_out_len = out_audio.shape[0]
    _working_out = out_audio[:min_len]
    _working_ref = ref_audio[:min_len]

    eps = 1e-12
    x = _working_out.astype(np.float64, copy=False)
    r = _working_ref.astype(np.float64, copy=False)

    # Mono reference for mask estimation
    r_m = np.mean(r, axis=1)

    # Peak envelope (~1 ms max)
    peak_win = max(8, int(sr * 0.001))
    peak_env = ndimage.maximum_filter1d(np.abs(r_m), size=peak_win, mode="nearest")

    # RMS envelope (~10 ms)
    rms_win = max(32, int(sr * 0.010))
    k = np.ones(rms_win, dtype=np.float64) / float(rms_win)
    rms_env = np.sqrt(np.convolve(r_m * r_m, k, mode="same") + eps)

    transientness = peak_env / (rms_env + eps)

    # Map transientness to mask [0..1]
    t0, t1 = 1.35, 2.40
    mask = np.clip((transientness - t0) / (t1 - t0 + eps), 0.0, 1.0)

    # Attack/Release smoothing on mask (time-domain 1-pole)
    a_ms, r_ms = 2.0, 40.0
    a = math.exp(-1.0 / (max(1.0, a_ms) * 0.001 * sr))
    b = math.exp(-1.0 / (max(1.0, r_ms) * 0.001 * sr))

    y = np.zeros_like(mask)
    prev = 0.0
    for i in range(mask.shape[0]):
        m = float(mask[i])
        if m > prev:
            prev = a * prev + (1.0 - a) * m
        else:
            prev = b * prev + (1.0 - b) * m
        y[i] = prev
    mask = np.clip(y, 0.0, 1.0)

    # Only blend where reference peaks exceed processed peaks (prevents "undoing" good improvements)
    p_m = np.mean(x, axis=1)
    peak_env_p = ndimage.maximum_filter1d(np.abs(p_m), size=peak_win, mode="nearest")
    boost_need = np.clip((peak_env - peak_env_p) / (peak_env + eps), 0.0, 1.0)

    m = (strength * mask * boost_need).astype(np.float64)
    if np.max(m) < 1e-6:
        return out_audio

    x = x * (1.0 - m[:, None]) + r * (m[:, None])
    # Restore original length if out_audio was longer
    if _orig_out_len > min_len:
        x = np.concatenate([x, out_audio[min_len:].astype(np.float64)], axis=0)
    return x

def _print_progress(step: int, total: int, label: str):
    bar_len = 28
    filled = int(bar_len * step / max(total, 1))
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"[{bar}] {step}/{total} {label}")

def _ensure_pip_package(import_name: str, pip_name: str, extra: Optional[str] = None) -> bool:
    """
    One-click helper: if a dependency is missing, attempt to pip install it automatically.
    Returns True if import succeeds after install, else False.
    """
    try:
        __import__(import_name)
        return True
    except Exception:
        pass

    pkg = pip_name + (extra or "")
    print(f"  ⬇️  Installing missing dependency: {pkg}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    except Exception as e:
        print(f"  ❌ pip install failed for {pkg}: {e}")
        return False

    try:
        __import__(import_name)
        return True
    except Exception as e:
        print(f"  ❌ Import still failing for {import_name}: {e}")
        return False

def _lowpass_filtfilt(audio: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Lowpass filter with gentler slope to avoid transition-band artifacts"""
    if cutoff_hz <= 0 or cutoff_hz >= 0.49 * sr:
        return audio
    # Use lower order for gentler slope (reduced from 8 to 4)
    sos = signal.butter(min(order, 4), cutoff_hz / (sr * 0.5), btype="lowpass", output="sos")
    y = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        y[:, ch] = signal.sosfiltfilt(sos, audio[:, ch]).astype(np.float64)
    return y

def _highpass_filtfilt(audio: np.ndarray, sr: int, cutoff_hz: float, order: int = 6) -> np.ndarray:
    if cutoff_hz <= 0:
        return audio
    sos = signal.butter(order, cutoff_hz / (sr * 0.5), btype="highpass", output="sos")
    y = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        y[:, ch] = signal.sosfiltfilt(sos, audio[:, ch]).astype(np.float64)
    return y

def _transient_enhance(audio: np.ndarray, sr: int, strength: float = 0.35) -> np.ndarray:
    """
    Transient restoration for AI music:
    - onset detection (librosa if available, else energy flux)
    - build a short, decaying gain envelope around onsets
    - apply mostly to mid/high content to avoid bass pumping
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-6:
        return audio

    # Split bands (Linkwitz-Riley-ish via cascaded Butterworth + filtfilt)
    x = audio.astype(np.float64)
    lo = _lowpass_filtfilt(x, sr, 150.0, order=6)
    mid = x - lo
    mid_lo = _lowpass_filtfilt(mid, sr, 4000.0, order=6)
    hi = mid - mid_lo
    mid = mid_lo

    # Mono for onset detection
    mono = np.mean(x, axis=1)

    # Onset strength
    if LIBROSA_AVAILABLE:
        # spectral flux style onset envelope
        hop = 256
        try:
            oenv = librosa.onset.onset_strength(y=mono.astype(np.float32), sr=sr, hop_length=hop)
            onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop, units="samples")
        except Exception:
            onsets = np.array([], dtype=np.int64)
    else:
        hop = 256
        # simple energy flux
        n = len(mono)
        frames = 1 + max(0, (n - hop) // hop)
        e = np.zeros(frames, dtype=np.float64)
        for i in range(frames):
            s = i * hop
            seg = mono[s:s+hop]
            e[i] = np.sum(seg * seg)
        flux = np.maximum(0.0, np.diff(e, prepend=e[0]))
        thr = np.mean(flux) + 2.0 * np.std(flux)
        idx = np.where(flux > thr)[0]
        onsets = (idx * hop).astype(np.int64)

    if onsets.size == 0:
        return audio

    # Build gain envelope: impulses -> exponential decay (attack very fast)
    env = np.zeros(len(mono), dtype=np.float64)
    env[onsets] = 1.0
    # decay ~ 35ms, and a tiny pre-emphasis window ~ 4ms
    decay_s = 0.035
    k_len = int(decay_s * sr)
    k_len = max(16, k_len)
    t = np.arange(k_len) / sr
    kernel = np.exp(-t / decay_s)
    env = signal.fftconvolve(env, kernel, mode="full")[:len(env)]
    env /= (np.max(env) + 1e-12)

    # shaped gain curve; cap to avoid harshness
    g = 1.0 + (0.65 * strength) * env
    g = np.clip(g, 1.0, 1.0 + 0.85 * strength)

    y = np.empty_like(x)

    # Apply to mid+hi with slight emphasis; leave lows mostly intact.
    for ch in range(x.shape[1]):
        y[:, ch] = lo[:, ch] + (mid[:, ch] * g) + (hi[:, ch] * (g ** 1.15))

    return y

def _deharsh_degrain(audio: np.ndarray, sr: int, amount: float = 0.5) -> np.ndarray:
    """
    Reduce "overdriven" AI roughness without dulling:
    - dynamic attenuation in 5-12 kHz when the band looks noise-like
      (high spectral flatness) and overly dominant.
    Works directly on the mix or on selected stems.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 1e-6:
        return audio

    n_fft = 2048
    hop = 256
    stft = GPUStft(n_fft=n_fft, hop_length=hop)

    y = np.zeros_like(audio, dtype=np.float64)

    # bands in Hz
    f_lo = 5000.0
    f_hi = 12000.0

    for ch in range(audio.shape[1]):
        X = stft.stft(audio[:, ch].astype(np.float64))
        mag = np.abs(X) + 1e-12
        phase = X / mag

        freqs = np.linspace(0, sr * 0.5, mag.shape[0])
        band = (freqs >= f_lo) & (freqs <= f_hi)

        # spectral flatness in band (per frame)
        band_mag = mag[band, :]
        geo = np.exp(np.mean(np.log(band_mag + 1e-12), axis=0))
        arith = np.mean(band_mag, axis=0) + 1e-12
        flat = np.clip(geo / arith, 0.0, 1.0)  # 1.0 = noise-like

        # band dominance vs total (per frame)
        total = np.mean(mag, axis=0) + 1e-12
        dom = np.clip(np.mean(band_mag, axis=0) / total, 0.0, 5.0)

        # gain: attenuate when flatness high and dominance high
        # thresholds tuned for AI shimmer/drive
        flat_w = np.clip((flat - 0.25) / 0.55, 0.0, 1.0)
        dom_w = np.clip((dom - 0.18) / 0.35, 0.0, 1.0)
        w = flat_w * dom_w

        # smooth weights to avoid flutter (release ~ 80ms)
        rel = int(0.08 * sr / hop)
        rel = max(2, rel)
        w_s = ndimage.uniform_filter1d(w, size=rel, mode="nearest")

        # attenuation up to ~6 dB (amount-scaled)
        att_db = -6.0 * amount * w_s
        g = 10.0 ** (att_db / 20.0)

        mag2 = mag.copy()
        mag2[band, :] *= g[None, :]

        X2 = mag2 * phase
        y[:, ch] = stft.istft(X2, length=audio.shape[0])

    return y

def _true_peak_limiter(audio: np.ndarray, sr: int, target_dbfs: float = -1.0, oversample: int = 4) -> np.ndarray:
    """
    Final TRUE-PEAK safety WITHOUT transient shaving.

    Behavior:
      - Estimate true peak via oversampling (polyphase resample).
      - If true peak exceeds target, apply the *minimum required* GLOBAL gain reduction.
      - No soft limiting / waveshaping here. (Those are what were "compressing" attacks.)
    """
    target = 10.0 ** (target_dbfs / 20.0)
    if target <= 0:
        return audio

    # Estimate true peak
    try:
        up = signal.resample_poly(audio, oversample, 1, axis=0)
        tp = float(np.max(np.abs(up)))
    except Exception:
        tp = float(np.max(np.abs(audio)))

    if tp <= target:
        return audio.astype(np.float64, copy=False)

    g = target / (tp + 1e-12)
    return (audio.astype(np.float64, copy=False) * g)


def _ultrasonic_cleanup(audio: np.ndarray, sr: int, cutoff_hz: float = 20000.0) -> np.ndarray:
    """
    Remove ultrasonic artifacts (like 22kHz noise) while preserving audible content.
    
    Uses a gentle lowpass with smooth rolloff starting at cutoff_hz.
    This cleans up artifacts from HFR synthesis and other processing
    without affecting the audible spectrum.
    """
    if sr <= 44100:
        # At 44.1kHz, Nyquist is 22.05kHz - not much room for ultrasonic noise
        # Just apply a very gentle rolloff
        cutoff_hz = min(cutoff_hz, sr * 0.45)
    
    if cutoff_hz >= sr * 0.49:
        return audio
    
    # Use frequency-domain processing for precise control
    n_fft = 4096
    hop = 1024
    stft = GPUStft(n_fft=n_fft, hop_length=hop)
    
    result = np.zeros_like(audio, dtype=np.float64)
    
    for ch in range(audio.shape[1]):
        spec = stft.stft(audio[:, ch].astype(np.float64))
        num_bins = spec.shape[0]
        
        # Calculate frequency per bin
        freq_per_bin = (sr / 2.0) / num_bins
        
        # Create smooth rolloff weights
        freqs = np.arange(num_bins) * freq_per_bin
        
        # Smooth rolloff: full pass below cutoff, gentle roll above
        # Using a cosine taper — start rolloff 2kHz below cutoff for a very wide, invisible transition
        weights = np.ones(num_bins)
        
        rolloff_start = max(cutoff_hz - 4000.0, cutoff_hz * 0.75)  # Start very early
        rolloff_end = sr * 0.48  # Stop just before Nyquist
        
        rolloff_mask = freqs > rolloff_start
        if np.any(rolloff_mask):
            rolloff_freqs = freqs[rolloff_mask]
            # Cosine rolloff
            rolloff_progress = (rolloff_freqs - rolloff_start) / (rolloff_end - rolloff_start + 1e-6)
            rolloff_progress = np.clip(rolloff_progress, 0, 1)
            rolloff_curve = 0.5 + 0.5 * np.cos(rolloff_progress * np.pi)
            weights[rolloff_mask] = rolloff_curve
        
        # Apply weights
        spec_filtered = spec * weights[:, np.newaxis]
        
        # Reconstruct
        result[:, ch] = stft.istft(spec_filtered, length=audio.shape[0])
    
    return result


# =========================
# Spectral Flux Correction — restore smeared transients in AI music
# =========================

def _spectral_flux_correction(audio: np.ndarray, sr: int,
                              strength: float = 0.65,
                              flux_threshold: float = 0.30) -> np.ndarray:
    """Sharpen transient attacks that AI generators smear across extra milliseconds.

    AI music generators produce magnitude-correct transients but spread the onset
    energy over 5-15 ms instead of the 1-3 ms a real instrument would have.
    This shows up as abnormally LOW spectral flux relative to the energy change.

    Algorithm:
      1. Compute per-frame spectral flux (positive half-wave rectified energy change).
      2. Compute per-frame total energy delta (how much the signal *should* be changing).
      3. Build a "smear mask" where flux/energy_delta is below threshold
         (frames that changed energy but without the spectral sharpness of a real attack).
      4. For those frames, apply short-window gain sharpening: boost the first 1-2 ms
         of the onset relative to the trailing 5-10 ms. This restores the "snap"
         without changing the overall level.

    Only operates on mid+high bands (>200 Hz) to avoid bass pumping.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength < 0.01:
        return audio

    n_fft = 2048
    hop = 512
    stft = GPUStft(n_fft=n_fft, hop_length=hop)
    eps = 1e-12

    # Work on copy
    x = audio.astype(np.float64).copy()

    # Split: keep bass clean, process mid+high
    try:
        sos_lo = signal.butter(4, 200.0, btype='lowpass', fs=sr, output='sos')
        x_lo = np.zeros_like(x)
        x_hi = np.zeros_like(x)
        for ch in range(x.shape[1]):
            x_lo[:, ch] = signal.sosfiltfilt(sos_lo, x[:, ch])
        x_hi = x - x_lo
    except Exception:
        x_lo = np.zeros_like(x)
        x_hi = x.copy()

    # Mono for analysis
    mono = np.mean(x_hi, axis=1) if x_hi.ndim == 2 else x_hi

    # STFT analysis
    S = stft.stft(mono)
    mag = np.abs(S) + eps
    num_bins, num_frames = mag.shape

    # Per-frame energy
    frame_energy = np.sum(mag ** 2, axis=0)
    frame_energy_db = 10.0 * np.log10(frame_energy + eps)

    # Spectral flux: sum of positive frame-to-frame magnitude changes
    flux = np.zeros(num_frames, dtype=np.float64)
    if num_frames > 1:
        diff = np.diff(mag, axis=1)
        flux[1:] = np.sum(np.maximum(diff, 0.0), axis=0)

    # Energy delta: absolute frame-to-frame energy change
    energy_delta = np.zeros(num_frames, dtype=np.float64)
    if num_frames > 1:
        energy_delta[1:] = np.maximum(0.0, np.diff(frame_energy_db))

    # Normalize
    flux_max = np.max(flux) + eps
    edelta_max = np.max(energy_delta) + eps
    flux_norm = flux / flux_max
    edelta_norm = energy_delta / edelta_max

    # Smear detection: frames where energy changed significantly but flux is low
    # (energy went up → should be a transient, but flux is suspiciously smooth)
    smear_mask = np.zeros(num_frames, dtype=np.float64)
    for f_idx in range(1, num_frames):
        if edelta_norm[f_idx] > 0.15:  # significant energy increase
            flux_ratio = flux_norm[f_idx] / (edelta_norm[f_idx] + eps)
            if flux_ratio < flux_threshold:
                # This frame is smeared: energy changed but spectrum didn't snap
                smear_mask[f_idx] = np.clip(1.0 - flux_ratio / flux_threshold, 0.0, 1.0)

    # Expand mask: protect 2 frames before and 4 frames after each detected onset
    try:
        smear_mask = ndimage.maximum_filter1d(smear_mask, size=6, origin=-2)
    except Exception:
        pass

    # Smooth the mask
    try:
        smear_mask = ndimage.gaussian_filter1d(
            smear_mask.astype(np.float32), sigma=1.0, mode='nearest'
        ).astype(np.float64)
    except Exception:
        pass

    # Convert frame-level mask to sample-level
    n_samples = x_hi.shape[0]
    smear_samples = np.zeros(n_samples, dtype=np.float64)
    for f_idx in range(num_frames):
        if smear_mask[f_idx] < 0.01:
            continue
        sample_start = f_idx * hop
        sample_end = min(n_samples, sample_start + n_fft)
        smear_samples[sample_start:sample_end] = np.maximum(
            smear_samples[sample_start:sample_end], smear_mask[f_idx]
        )

    # Build sharpening gain: boost the first ~1.5ms, attenuate the next ~8ms
    # This creates a "snappier" attack envelope
    attack_samples = max(8, int(sr * 0.0015))   # 1.5 ms
    body_samples = max(32, int(sr * 0.008))      # 8 ms

    # Fast peak envelope of the mid+high signal
    mono_abs = np.abs(mono) + eps
    peak_env = ndimage.maximum_filter1d(mono_abs, size=attack_samples, mode='nearest')
    rms_env = np.sqrt(
        ndimage.uniform_filter1d(mono ** 2, size=body_samples, mode='nearest') + eps
    )

    # Crest factor: where peak >> rms, the attack is already sharp → skip
    # Where peak ≈ rms, the attack is rounded → sharpen
    crest = peak_env / (rms_env + eps)
    # Normalize so low crest (rounded attacks) → high correction need
    crest_norm = np.clip(crest, 1.0, 5.0)
    sharpening_need = 1.0 - (crest_norm - 1.0) / 4.0  # 1=round, 0=sharp
    sharpening_need = np.clip(sharpening_need, 0.0, 1.0)

    # Combined gain adjustment
    gain_adjust = 1.0 + (0.35 * strength * smear_samples * sharpening_need)
    gain_adjust = np.clip(gain_adjust, 1.0, 1.0 + 0.5 * strength)

    # Smooth the gain to avoid clicks
    try:
        gain_adjust = ndimage.gaussian_filter1d(
            gain_adjust.astype(np.float32), sigma=max(2, attack_samples // 4),
            mode='nearest'
        ).astype(np.float64)
    except Exception:
        pass

    # Apply to mid+high
    for ch in range(x_hi.shape[1]):
        x_hi[:, ch] *= gain_adjust

    result = x_lo + x_hi

    # Preserve peak
    orig_peak = np.max(np.abs(audio)) + eps
    res_peak = np.max(np.abs(result)) + eps
    if res_peak > orig_peak:
        result *= (orig_peak / res_peak)

    return result


# =========================
# CQT-Domain Harmonic Smoothing
# =========================

def _cqt_harmonic_smoothing(audio: np.ndarray, sr: int,
                            strength: float = 0.20,
                            bins_per_octave: int = 48) -> np.ndarray:
    """Enforce harmonic-decay consistency in the Constant-Q domain.

    IMPORTANT: Only operates on the SYNTHESIZED high-frequency region (above the
    detected bandwidth cutoff), NOT on the original content below it. The original
    sub-cutoff material is already clean and must not be blurred.

    Algorithm:
      1. Detect the bandwidth cutoff (where original content ends).
      2. Forward CQT with fine resolution (48 bins/octave).
      3. In the CQT bins ABOVE the cutoff, measure how jagged the harmonic
         amplitude envelope is — compute a smoothed version and only correct
         the *deviation* from smooth, leaving the absolute shape intact.
      4. NO temporal smoothing (that causes visible blur in spectral analyzers).
      5. Gentle blend (default 20%) confined to the HF-extension region.

    The result is more coherent overtone decay in synthesized HF without any
    perceptible blur in the original content or in transient detail.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength < 0.01:
        return audio

    if not LIBROSA_AVAILABLE:
        return audio

    import librosa

    eps = 1e-12
    result = audio.copy().astype(np.float64)
    orig_peak = np.max(np.abs(audio)) + eps

    n_octaves = max(4, int(np.log2(max(2000, sr / 2.0) / 32.5)))
    n_bins = n_octaves * bins_per_octave
    fmin = 32.7  # C1

    # ── Detect bandwidth cutoff (same method as HFPv2/AnalogVitality) ──
    mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
    stft_det = GPUStft(n_fft=4096, hop_length=1024)
    try:
        S_det = stft_det.stft(mono.astype(np.float64))
        mag_det = np.abs(S_det) + eps
        e_avg = np.mean(mag_det, axis=1)
        e_db = 20.0 * np.log10(e_avg + eps)
        peak_db = float(np.max(e_db))
        bin_hz = sr / 4096.0
        cutoff_hz = sr * 0.45  # fallback: near Nyquist
        for b in range(int(2000 / bin_hz), int(sr * 0.48 / bin_hz)):
            if b >= len(e_db) - 48:
                break
            tail = e_db[b:b + 48]
            if tail.size >= 24 and np.median(tail) < (peak_db - 34.0):
                cutoff_hz = float(b * bin_hz)
                break
    except Exception:
        cutoff_hz = sr * 0.45

    # If cutoff is near Nyquist, nothing synthesized → skip
    if cutoff_hz >= sr * 0.42:
        return audio

    for ch in range(audio.shape[1] if audio.ndim == 2 else 1):
        x = audio[:, ch] if audio.ndim == 2 else audio
        x = x.astype(np.float32)

        try:
            C = librosa.cqt(x, sr=sr, fmin=fmin, n_bins=n_bins,
                            bins_per_octave=bins_per_octave,
                            hop_length=512)
            mag = np.abs(C) + eps
            phase = np.angle(C)

            cqt_freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin,
                                                 bins_per_octave=bins_per_octave)

            # Only operate above the detected cutoff (with a transition zone)
            cutoff_bin = np.searchsorted(cqt_freqs, cutoff_hz)
            transition_bins = max(4, bins_per_octave // 6)  # ~⅙ octave fade-in
            start_bin = max(0, cutoff_bin - transition_bins)

            if start_bin >= n_bins - 8:
                continue

            # ── Deviation-based smoothing (NOT raw magnitude blur) ──
            # Compute smooth reference for the HF region, then correct only
            # the per-frame deviation from that reference. This preserves
            # the absolute spectral shape while taming jagged artifacts.
            mag_hf = mag[start_bin:].copy()

            # Single-pass mild frequency smoothing (sigma=1.5 bins = 1/32 octave)
            try:
                mag_ref = ndimage.gaussian_filter1d(
                    mag_hf.astype(np.float32), sigma=1.5, axis=0,
                    mode='nearest'
                ).astype(np.float64)
            except Exception:
                mag_ref = mag_hf.copy()

            # Deviation: how much each bin deviates from the smooth reference
            # Ratio > 1 means spike, < 1 means dip
            ratio = mag_hf / (mag_ref + eps)

            # Only correct outlier deviations (>2 dB spike or >3 dB dip)
            # This leaves well-behaved harmonics untouched
            ratio_db = 20.0 * np.log10(ratio + eps)
            correction = np.ones_like(ratio)
            spike_mask = ratio_db > 2.0
            dip_mask = ratio_db < -3.0
            # Attenuate spikes toward reference
            correction[spike_mask] = (mag_ref[spike_mask] / (mag_hf[spike_mask] + eps))
            # Boost dips toward reference (gently)
            correction[dip_mask] = (mag_ref[dip_mask] / (mag_hf[dip_mask] + eps))
            # Soft-limit correction to avoid large jumps
            correction = np.clip(correction, 0.5, 2.0)

            # Build per-bin blend that fades from 0 at start_bin to strength at cutoff_bin
            blend_hf = np.zeros(mag_hf.shape[0], dtype=np.float64)
            for b in range(mag_hf.shape[0]):
                abs_bin = b + start_bin
                if abs_bin >= cutoff_bin:
                    blend_hf[b] = strength
                elif abs_bin >= start_bin:
                    t = (abs_bin - start_bin) / max(1, cutoff_bin - start_bin)
                    blend_hf[b] = strength * (0.5 - 0.5 * np.cos(np.pi * t))

            # Apply: blend between original and corrected
            mag_corrected = mag_hf * (1.0 + blend_hf[:, None] * (correction - 1.0))

            # Reconstruct
            mag_out = mag.copy()
            mag_out[start_bin:] = mag_corrected

            C_out = mag_out * np.exp(1j * phase)
            y = librosa.icqt(C_out, sr=sr, fmin=fmin,
                             bins_per_octave=bins_per_octave,
                             hop_length=512)

            n_samp = len(x)
            if len(y) > n_samp:
                y = y[:n_samp]
            elif len(y) < n_samp:
                y = np.pad(y, (0, n_samp - len(y)))

            if audio.ndim == 2:
                result[:, ch] = y.astype(np.float64)
            else:
                result = y.astype(np.float64)

        except Exception as e:
            print(f"      CQT smoothing failed for ch {ch}: {e}")
            continue

    # Preserve peak
    res_peak = np.max(np.abs(result)) + eps
    if res_peak > orig_peak:
        result *= (orig_peak / res_peak)

    return result


# =========================
# MP-SENet Phase Restoration (Black-Box)
# =========================

def _run_mpsenet_phase_restore(audio: np.ndarray, sr: int,
                               cfg) -> np.ndarray:
    """Run MP-SENet for explicit phase spectrum restoration.

    MP-SENet (Magnitude and Phase Speech Enhancement Network) jointly optimizes
    magnitude and phase using parallel estimation with anti-wrapping losses.
    At only 6.6M parameters it's lightweight and fast.

    Integration strategy:
      - Clone the repo into models/mpsenet/ if not present
      - Run inference as a subprocess (like Apollo)
      - Falls back gracefully if unavailable

    If MP-SENet is not installed, this is a no-op.
    """
    if not getattr(cfg, 'ENABLE_MPSENET', True):
        return audio

    mpsenet_dir = getattr(cfg, 'MPSENET_DIR',
                          os.path.join(_SCRIPT_DIR, "models", "mpsenet"))

    # Check if MP-SENet repo exists
    inference_script = os.path.join(mpsenet_dir, "inferencer.py")
    if not os.path.isfile(inference_script):
        # Try alternative script name
        inference_script = os.path.join(mpsenet_dir, "inference.py")
    if not os.path.isfile(inference_script):
        # Try to find any main inference entry point
        for candidate in ["enhance.py", "run.py", "test.py"]:
            p = os.path.join(mpsenet_dir, candidate)
            if os.path.isfile(p):
                inference_script = p
                break
        else:
            # MP-SENet not installed — try cloning
            repo_url = getattr(cfg, 'MPSENET_REPO_URL',
                               'https://github.com/yxlu-0102/MP-SENet.git')
            print(f"      MP-SENet not found at {mpsenet_dir}")
            print(f"      To enable: git clone {repo_url} {mpsenet_dir}")
            print(f"      Skipping phase restoration (pipeline continues without it)")
            return audio

    # MP-SENet expects 16kHz mono input — we process per-channel
    mpsenet_sr = 16000
    orig_sr = int(sr)
    orig_audio = audio.copy()
    orig_peak = float(np.max(np.abs(audio))) + 1e-12

    print(f"      Running MP-SENet phase restoration...")

    result_channels = []
    n_ch = audio.shape[1] if audio.ndim == 2 else 1

    for ch in range(n_ch):
        x = audio[:, ch] if audio.ndim == 2 else audio
        x = x.astype(np.float64)

        # Resample to 16kHz for MP-SENet
        if orig_sr != mpsenet_sr:
            x_16k = resample_gpu(x[:, np.newaxis] if x.ndim == 1 else x,
                                 orig_sr, mpsenet_sr)
            if x_16k.ndim == 2:
                x_16k = x_16k[:, 0]
        else:
            x_16k = x.copy()

        # Write to temp file
        try:
            with tempfile.TemporaryDirectory() as td:
                in_wav = os.path.join(td, "input.wav")
                out_dir = os.path.join(td, "output")
                os.makedirs(out_dir, exist_ok=True)

                sf.write(in_wav, x_16k.astype(np.float32), mpsenet_sr, subtype="FLOAT")

                # Run MP-SENet inference
                cmd = [sys.executable, inference_script,
                       "--test_dir", td,
                       "--output_dir", out_dir]

                result_sub = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300,
                    cwd=mpsenet_dir,
                )

                if result_sub.returncode != 0:
                    print(f"      ⚠️  MP-SENet returned code {result_sub.returncode}; skipping ch {ch}")
                    result_channels.append(x)
                    continue

                # Read output
                out_files = sorted(glob.glob(os.path.join(out_dir, "**", "*.wav"),
                                              recursive=True))
                if not out_files:
                    print(f"      ⚠️  MP-SENet produced no output; skipping ch {ch}")
                    result_channels.append(x)
                    continue

                enhanced, enh_sr = sf.read(out_files[0])
                enhanced = enhanced.astype(np.float64)
                if enhanced.ndim == 2:
                    enhanced = enhanced[:, 0]

                # Resample back to original SR
                if int(enh_sr) != orig_sr:
                    enhanced = resample_gpu(
                        enhanced[:, np.newaxis], int(enh_sr), orig_sr)[:, 0]

                # Length match
                orig_len = len(x)
                if len(enhanced) > orig_len:
                    enhanced = enhanced[:orig_len]
                elif len(enhanced) < orig_len:
                    enhanced = np.pad(enhanced, (0, orig_len - len(enhanced)))

                # Blend: use MP-SENet's phase but preserve original magnitude character
                # 60% enhanced / 40% original gives phase improvement without timbre shift
                blended = 0.6 * enhanced + 0.4 * x

                result_channels.append(blended)

        except subprocess.TimeoutExpired:
            print(f"      ⚠️  MP-SENet timed out; skipping ch {ch}")
            result_channels.append(x)
        except Exception as e:
            print(f"      ⚠️  MP-SENet failed: {e}; skipping ch {ch}")
            result_channels.append(x)

    # Reconstruct
    if audio.ndim == 2:
        result = np.column_stack(result_channels)
    else:
        result = result_channels[0]

    # Preserve peak
    res_peak = float(np.max(np.abs(result))) + 1e-12
    if res_peak > orig_peak:
        result *= (orig_peak / res_peak)

    print(f"      ✓ MP-SENet phase restoration complete")
    return result


def _hf_naturalize(audio: np.ndarray, sr: int,
                   transition_hz: float = 11000.0,
                   diffusion_strength: float = 1.0,
                   noise_floor_db: float = -45.0,
                   phase_diffusion_amount: float = 0.6) -> np.ndarray:
    """Naturalize the HF spectrum with FREQUENCY-PROPORTIONAL blur.

    Three treatments above transition_hz, with intensity that RAMPS with
    frequency (power curve t^1.5) — very light near the transition (preserving
    good restoration output), heavy near Nyquist (masking synthetic content):

    1. SPECTRAL DIFFUSION — frequency-axis Gaussian blur + transient-aware
       temporal smoothing.  Strength is ~8% at the transition and ramps to
       100% at Nyquist via a power curve that stays light longer.

    2. SHAPED NOISE FLOOR — fills dead-silent gaps between harmonics with
       spectrally-shaped noise.

    3. PHASE DIFFUSION — partially randomize phase, gated by transient mask
       and frequency ramp.

    Crossfades in from transition_hz * 0.80 to transition_hz * 1.20.
    """
    if diffusion_strength < 0.01:
        return audio

    n_fft = 4096
    hop = 1024
    stft = GPUStft(n_fft=n_fft, hop_length=hop)
    eps = 1e-10

    result = np.zeros_like(audio, dtype=np.float64)

    num_bins = n_fft // 2 + 1
    freq_per_bin = (sr / 2.0) / num_bins
    freqs = np.arange(num_bins) * freq_per_bin
    nyquist = sr / 2.0

    # Wide crossfade: starts at 80% of transition, fully on at 120%
    # At 15kHz transition: ramp from 12kHz to 18kHz — very gradual
    fade_low = transition_hz * 0.80
    fade_high = transition_hz * 1.20
    blend = np.zeros(num_bins, dtype=np.float64)
    for b in range(num_bins):
        if freqs[b] <= fade_low:
            blend[b] = 0.0
        elif freqs[b] >= fade_high:
            blend[b] = 1.0
        else:
            t = (freqs[b] - fade_low) / (fade_high - fade_low)
            blend[b] = 0.5 - 0.5 * np.cos(np.pi * t)

    # FREQUENCY-PROPORTIONAL SCALING: blur ramps from light near transition
    # to full strength at Nyquist.  This preserves AudioSR's good output in
    # the 15-17 kHz range while still masking synthetic content near Nyquist.
    # Uses a power curve (t^1.5) so it stays lighter longer, then ramps up.
    RAMP_FLOOR = 0.30  # 30% at transition (was 0.08)
    freq_ramp = np.ones(num_bins, dtype=np.float64)
    if nyquist > fade_high:
        for b in range(num_bins):
            if freqs[b] >= fade_low and freqs[b] < fade_high:
                # In the crossfade zone: use floor
                freq_ramp[b] = RAMP_FLOOR
            elif freqs[b] >= fade_high:
                t = (freqs[b] - fade_high) / (nyquist - fade_high)
                t = min(1.0, t)
                # Power curve: stays low longer, ramps up near Nyquist
                freq_ramp[b] = RAMP_FLOOR + (1.0 - RAMP_FLOOR) * (t ** 1.5)

    blend_scaled = blend * freq_ramp * min(1.0, diffusion_strength)

    # Heavy frequency diffusion sigma (this is the good part — keep it)
    freq_sigma = max(3.0, 15.0 * diffusion_strength)

    # Noise floor
    noise_amp = 10.0 ** (noise_floor_db / 20.0)

    # Phase diffusion amount
    phase_diffusion = min(1.0, phase_diffusion_amount * diffusion_strength)

    for ch in range(audio.shape[1] if audio.ndim == 2 else 1):
        x = audio[:, ch] if audio.ndim == 2 else audio
        S = stft.stft(x.astype(np.float64))
        mag = np.abs(S)
        phase = np.angle(S)
        num_b, num_frames = mag.shape

        # =============================================
        # TRANSIENT DETECTION — spectral flux onset mask
        # =============================================
        # Compute broadband energy per frame
        frame_energy = np.sum(mag ** 2, axis=0)  # (num_frames,)
        frame_energy_db = 10.0 * np.log10(frame_energy + eps)

        # Spectral flux: positive half-wave rectified frame-to-frame increase
        flux = np.zeros(num_frames, dtype=np.float64)
        if num_frames > 1:
            diff = np.diff(frame_energy_db)
            flux[1:] = np.maximum(0.0, diff)

        # Smooth the flux slightly to get ~2-3 frame attack window
        try:
            flux_smooth = ndimage.gaussian_filter1d(
                flux.astype(np.float32), sigma=1.0, mode='nearest'
            ).astype(np.float64)
        except Exception:
            flux_smooth = flux

        # Normalize flux to 0..1 (0 = sustained, 1 = strong onset)
        flux_max = np.max(flux_smooth) + eps
        flux_norm = flux_smooth / flux_max

        # Transient mask: 1.0 = transient (no time blur), 0.0 = sustained (full time blur)
        # Threshold: anything above 25% of max flux is considered transient-ish
        transient_threshold = 0.25
        transient_mask = np.clip(flux_norm / transient_threshold, 0.0, 1.0)

        # Expand transients slightly: each onset protects ~3 frames after it too
        # (so the attack body isn't smeared, just the tail)
        try:
            # Max-filter forward to protect frames after onsets
            from scipy.ndimage import maximum_filter1d
            transient_mask = maximum_filter1d(transient_mask, size=5, origin=-2)
        except Exception:
            pass

        # sustain_mask: inverse — where it's safe to apply temporal smoothing
        sustain_mask = 1.0 - transient_mask  # (num_frames,)

        # =============================================
        # 1. SPECTRAL DIFFUSION — freq-heavy, time-transient-aware
        # =============================================

        # PASS A: frequency-only blur (2 passes, heavy)
        mag_freq_blurred = mag.copy()
        for _pass in range(2):
            try:
                mag_freq_blurred = ndimage.gaussian_filter1d(
                    mag_freq_blurred.astype(np.float32), sigma=freq_sigma,
                    axis=0, mode='nearest'
                ).astype(np.float64)
            except Exception:
                pass

        # PASS B: temporal blur of the freq-blurred result (gentle sigma)
        time_sigma = max(1.0, 1.5 * diffusion_strength)
        try:
            mag_time_blurred = ndimage.gaussian_filter1d(
                mag_freq_blurred.astype(np.float32), sigma=time_sigma,
                axis=1, mode='nearest'
            ).astype(np.float64)
        except Exception:
            mag_time_blurred = mag_freq_blurred.copy()

        # Blend between freq-only (at transients) and freq+time (at sustains)
        # This preserves temporal sharpness at attacks while smoothing sustains
        mag_diffused = (mag_freq_blurred * transient_mask[None, :] +
                        mag_time_blurred * sustain_mask[None, :])

        # Blend with original based on frequency crossfade
        mag_nat = mag.copy()
        for b in range(num_b):
            if blend_scaled[b] > 0.001:
                mag_nat[b, :] = (mag[b, :] * (1.0 - blend_scaled[b]) +
                                 mag_diffused[b, :] * blend_scaled[b])

        # =============================================
        # 2. SHAPED NOISE FLOOR — fill gaps, modulated by frame energy
        # =============================================
        if noise_floor_db > -120.0:
            # Wide local energy envelope
            try:
                local_env = ndimage.gaussian_filter1d(
                    mag_nat.astype(np.float32), sigma=20.0, axis=0, mode='nearest'
                ).astype(np.float64)
                # Gentle time smoothing for envelope reference only (not the signal)
                local_env = ndimage.gaussian_filter1d(
                    local_env.astype(np.float32), sigma=2.0, axis=1, mode='nearest'
                ).astype(np.float64)
            except Exception:
                local_env = mag_nat.copy()

            # Per-frame energy ratio: modulate noise by how loud this frame is
            # (quiet frames = much less noise fill)
            nf_frame_energy = np.sum(mag[:, :] ** 2, axis=0)
            nf_peak_energy = np.max(nf_frame_energy) + eps
            nf_energy_ratio = np.sqrt(nf_frame_energy / nf_peak_energy)
            try:
                nf_energy_ratio = ndimage.uniform_filter1d(
                    nf_energy_ratio.astype(np.float32), size=3, mode='nearest'
                ).astype(np.float64)
            except Exception:
                pass

            np.random.seed(137 + ch)
            noise_mag = np.abs(np.random.randn(num_b, num_frames)) * noise_amp

            env_max = np.max(local_env) + eps
            noise_shaped = noise_mag * (local_env / env_max)

            # Modulate by frame energy: quiet frames get proportionally less noise
            noise_shaped *= nf_energy_ratio[None, :]

            # Smooth noise in frequency only (NOT time — preserves transient structure)
            try:
                noise_shaped = ndimage.gaussian_filter1d(
                    noise_shaped.astype(np.float32),
                    sigma=4.0, axis=0, mode='nearest'
                ).astype(np.float64)
            except Exception:
                pass

            for b in range(num_b):
                if blend_scaled[b] > 0.01:
                    threshold = local_env[b, :] * 0.5
                    gap_mask = mag_nat[b, :] < (threshold + eps)
                    fill_level = noise_shaped[b, :] * blend_scaled[b] * 1.5
                    mag_nat[b, gap_mask] = np.maximum(mag_nat[b, gap_mask], fill_level[gap_mask])

        # =============================================
        # 3. PHASE DIFFUSION — transient-gated
        # =============================================
        phase_nat = phase.copy()
        if phase_diffusion > 0.01:
            np.random.seed(2049 + ch)
            random_phase = np.random.uniform(-np.pi, np.pi, size=phase.shape)

            # Smooth random phase in frequency only (keeps time-axis sharp)
            try:
                rp_sin = ndimage.gaussian_filter1d(
                    np.sin(random_phase).astype(np.float32), sigma=3.0,
                    axis=0, mode='wrap'
                )
                rp_cos = ndimage.gaussian_filter1d(
                    np.cos(random_phase).astype(np.float32), sigma=3.0,
                    axis=0, mode='wrap'
                )
                random_phase = np.arctan2(rp_sin, rp_cos).astype(np.float64)
            except Exception:
                pass

            for b in range(num_b):
                pd_base = blend_scaled[b] * phase_diffusion
                if pd_base > 0.001:
                    # Gate by sustain mask: full phase diffusion during sustains,
                    # near-zero at transients (keeps attacks phase-coherent)
                    pd_per_frame = pd_base * sustain_mask
                    orig_x = np.cos(phase[b, :]) * (1.0 - pd_per_frame) + np.cos(random_phase[b, :]) * pd_per_frame
                    orig_y = np.sin(phase[b, :]) * (1.0 - pd_per_frame) + np.sin(random_phase[b, :]) * pd_per_frame
                    phase_nat[b, :] = np.arctan2(orig_y, orig_x)

        # Reconstruct
        S_out = mag_nat * np.exp(1j * phase_nat)
        y = stft.istft(S_out, length=len(x))

        if audio.ndim == 2:
            result[:, ch] = y
        else:
            result = y

    # Preserve original peak
    orig_peak = np.max(np.abs(audio)) + eps
    res_peak = np.max(np.abs(result)) + eps
    if res_peak > orig_peak:
        result *= (orig_peak / res_peak)

    return result


def _hf_seam_boost(audio: np.ndarray, sr: int,
                   seam_hz: float = 0.0,
                   peak_boost_db: float = 8.0,
                   rolloff_db_per_oct: float = 6.0,
                   onset_width_hz: float = 2000.0) -> np.ndarray:
    """Boost the HF region with a curve that peaks at the seam and tilts down.

    The naturalization step smooths out synthetic texture but can leave the
    generated region visibly darker than the original content below it.  This
    function applies a spectral gain curve shaped like a tilted shelf:

        gain
         ^
    peak |      /‾‾‾-._
         |     /        ‾‾-._
    0 dB |----/               ‾‾-.__
         +------|---------|---------> freq
              onset     seam      Nyquist

    - Below (seam - onset_width): 0 dB (untouched)
    - Ramps up via raised-cosine over onset_width to reach peak_boost_db at seam
    - Above seam: decays at rolloff_db_per_oct (e.g. -6 dB/oct), mimicking
      the natural spectral tilt of real recordings
    - Never goes below 0 dB (no attenuation, only boost)

    If seam_hz <= 0, auto-detects the bandwidth cliff from the spectrum.

    Applied as a static gain curve in STFT domain — no transient distortion
    because it's a fixed EQ shape (same as any shelf filter, but with more
    precise spectral control).
    """
    if peak_boost_db < 0.1:
        return audio

    n_fft = 4096
    hop = 1024
    stft_engine = GPUStft(n_fft=n_fft, hop_length=hop)
    eps = 1e-10

    num_bins = n_fft // 2 + 1
    freq_per_bin = (sr / 2.0) / num_bins
    freqs = np.arange(num_bins) * freq_per_bin
    nyquist = sr / 2.0

    # --- Auto-detect seam if not specified ---
    if seam_hz <= 0:
        if audio.ndim == 2:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio
        S_detect = stft_engine.stft(mono.astype(np.float64))
        mag_detect = np.abs(S_detect)
        avg_energy = np.mean(mag_detect, axis=1) + eps
        avg_db = 20.0 * np.log10(avg_energy)

        # Smooth heavily to find the macro shape
        try:
            avg_db_smooth = ndimage.gaussian_filter1d(
                avg_db.astype(np.float32), sigma=8.0, mode='nearest'
            ).astype(np.float64)
        except Exception:
            avg_db_smooth = avg_db

        peak_db = float(np.max(avg_db_smooth))

        # Find the steepest drop between 8kHz and Nyquist-2kHz
        min_bin = max(1, int(8000.0 / freq_per_bin))
        max_bin = min(num_bins - 2, int((nyquist - 2000.0) / freq_per_bin))

        gradient = np.diff(avg_db_smooth)
        search_region = gradient[min_bin:max_bin]
        if search_region.size > 0:
            steepest = int(np.argmin(search_region)) + min_bin
            seam_hz = float(freqs[steepest])
        else:
            seam_hz = 14000.0

        # Clamp to reasonable range
        seam_hz = float(np.clip(seam_hz, 8000.0, nyquist - 2000.0))
        print(f"    Seam boost: auto-detected seam at {seam_hz:.0f} Hz")
    else:
        print(f"    Seam boost: using configured seam at {seam_hz:.0f} Hz")

    # --- Build the gain curve (in dB) ---
    gain_db = np.zeros(num_bins, dtype=np.float64)

    onset_start_hz = max(0.0, seam_hz - onset_width_hz)
    onset_start_bin = int(onset_start_hz / freq_per_bin)
    seam_bin = int(seam_hz / freq_per_bin)
    seam_bin = int(np.clip(seam_bin, 1, num_bins - 2))
    onset_start_bin = int(np.clip(onset_start_bin, 0, seam_bin - 1))

    # Onset ramp: 0 dB -> peak_boost_db via raised cosine
    if seam_bin > onset_start_bin:
        n_onset = seam_bin - onset_start_bin
        t = np.linspace(0.0, 1.0, n_onset, dtype=np.float64)
        ramp = 0.5 - 0.5 * np.cos(np.pi * t)
        gain_db[onset_start_bin:seam_bin] = ramp * peak_boost_db

    # At the seam: peak boost
    gain_db[seam_bin] = peak_boost_db

    # Above seam: decay at rolloff_db_per_oct
    # dB loss = rolloff_db_per_oct * log2(f / seam_hz)
    for b in range(seam_bin + 1, num_bins):
        f = freqs[b]
        if f > seam_hz and seam_hz > 0:
            octaves_above = np.log2(f / seam_hz)
            gain_db[b] = peak_boost_db - rolloff_db_per_oct * octaves_above
        else:
            gain_db[b] = peak_boost_db

    # Floor at 0 dB (never attenuate, only boost)
    gain_db = np.maximum(gain_db, 0.0)

    # Smooth the gain curve to avoid any sharp edges
    try:
        gain_db = ndimage.gaussian_filter1d(
            gain_db.astype(np.float32), sigma=3.0, mode='nearest'
        ).astype(np.float64)
        gain_db = np.maximum(gain_db, 0.0)
    except Exception:
        pass

    # Convert to linear gain
    gain_linear = 10.0 ** (gain_db / 20.0)

    print(f"    Seam boost: +{peak_boost_db:.1f} dB at {seam_hz:.0f} Hz, "
          f"-{rolloff_db_per_oct:.1f} dB/oct rolloff")

    # --- Apply in STFT domain ---
    result = np.zeros_like(audio, dtype=np.float64)

    for ch in range(audio.shape[1] if audio.ndim == 2 else 1):
        x = audio[:, ch] if audio.ndim == 2 else audio
        S = stft_engine.stft(x.astype(np.float64))

        S_boosted = S * gain_linear[:S.shape[0], np.newaxis]

        y = stft_engine.istft(S_boosted, length=len(x))

        if audio.ndim == 2:
            result[:, ch] = y
        else:
            result = y

    return result


def _dynamic_seam_bridge(audio: np.ndarray, sr: int, cfg) -> np.ndarray:
    """Dynamic, energy-following seam bridge.

    Per STFT frame, this function:

    1. DETECTS the seam — finds the steepest spectral drop-off in the
       11–18 kHz search window (the "cliff" between natural and synthetic).

    2. PLACES a Gaussian bell boost just above the detected seam
       (offset by ~600 Hz), acting as a smooth gradient that bridges
       the gap between the original HF tail and the AudioSR synthesis.

    3. MODULATES the boost by the energy in the 11.5–14 kHz reference
       band: frames with bright, energetic HF get the full 9 dB boost;
       frames where the highs are naturally weak get scaled down —
       preventing a static bright line from appearing on the spectrogram.

    Because both the center frequency and the gain vary per frame, the
    boost "breathes" with the music and tracks the seam as it drifts.
    """
    peak_db     = float(getattr(cfg, 'DYNAMIC_SEAM_DB', 9.0))
    Q           = float(getattr(cfg, 'DYNAMIC_SEAM_Q', 9.0))
    search_lo   = float(getattr(cfg, 'DYNAMIC_SEAM_SEARCH_LO', 11000.0))
    search_hi   = float(getattr(cfg, 'DYNAMIC_SEAM_SEARCH_HI', 18000.0))
    offset_hz   = float(getattr(cfg, 'DYNAMIC_SEAM_OFFSET_HZ', 600.0))
    ref_lo      = float(getattr(cfg, 'DYNAMIC_SEAM_REF_LO', 11500.0))
    ref_hi      = float(getattr(cfg, 'DYNAMIC_SEAM_REF_HI', 14000.0))
    gain_min    = float(getattr(cfg, 'DYNAMIC_SEAM_GAIN_MIN', 0.20))
    gain_max    = float(getattr(cfg, 'DYNAMIC_SEAM_GAIN_MAX', 1.40))

    if peak_db < 0.1:
        return audio

    # STFT params — 4096 gives ~11.7 Hz/bin resolution at 48 kHz
    n_fft = 4096
    hop   = 1024
    stft_engine = GPUStft(n_fft=n_fft, hop_length=hop)
    eps = 1e-10

    num_bins = n_fft // 2 + 1
    bin_hz   = (sr / 2.0) / num_bins
    freqs    = np.arange(num_bins) * bin_hz

    # Pre-compute bin ranges
    search_lo_bin = max(1, int(search_lo / bin_hz))
    search_hi_bin = min(num_bins - 2, int(search_hi / bin_hz))
    ref_lo_bin    = max(1, int(ref_lo / bin_hz))
    ref_hi_bin    = min(num_bins - 1, int(ref_hi / bin_hz))
    offset_bins   = max(1, int(offset_hz / bin_hz))

    # Pre-compute bin index array (used in Gaussian calculation)
    bin_indices = np.arange(num_bins, dtype=np.float64)

    # Soft onset ramp: cosine fade into the search region
    # Width is proportional to the search range (not a fixed 1kHz)
    onset_width_hz = max(300.0, (search_lo - 1000.0) * 0.15)
    onset_lo_bin = max(0, int((search_lo - onset_width_hz) / bin_hz))
    onset_mask = np.ones(num_bins, dtype=np.float64)
    onset_mask[:onset_lo_bin] = 0.0
    if onset_lo_bin < search_lo_bin:
        n_ramp = search_lo_bin - onset_lo_bin
        onset_mask[onset_lo_bin:search_lo_bin] = 0.5 - 0.5 * np.cos(
            np.pi * np.linspace(0.0, 1.0, n_ramp))

    result = np.zeros_like(audio, dtype=np.float64)
    seam_hz_report = []  # for logging

    for ch in range(audio.shape[1] if audio.ndim == 2 else 1):
        x = audio[:, ch] if audio.ndim == 2 else audio
        S = stft_engine.stft(x.astype(np.float64))
        mag   = np.abs(S) + eps
        phase = np.angle(S)
        _, num_frames = mag.shape

        # ─── 1. Per-frame seam detection ───────────────────────────
        # Smooth magnitude along freq axis to find macro spectral shape
        mag_smooth = ndimage.gaussian_filter1d(
            mag.astype(np.float32), sigma=8.0, axis=0, mode='nearest'
        ).astype(np.float64)

        # Spectral gradient in dB (derivative along frequency)
        mag_db = 20.0 * np.log10(mag_smooth + eps)
        grad   = np.diff(mag_db, axis=0)  # shape (num_bins-1, num_frames)

        # Steepest negative gradient in search window = seam location
        search_grad = grad[search_lo_bin:search_hi_bin, :]
        seam_bins_raw = np.argmin(search_grad, axis=0) + search_lo_bin

        # Temporal smoothing: median filter (outlier-robust) → Gaussian
        # Keep fast enough to follow attacks (~60ms window)
        seam_f = seam_bins_raw.astype(np.float64)
        try:
            seam_f = ndimage.median_filter(seam_f, size=5, mode='nearest')
            seam_f = ndimage.gaussian_filter1d(
                seam_f.astype(np.float32), sigma=3.0, mode='nearest'
            ).astype(np.float64)
        except Exception:
            pass

        seam_hz_report.append(float(np.median(seam_f) * bin_hz))

        # ─── 2. Per-frame reference energy ─────────────────────────
        ref_energy = np.sum(mag[ref_lo_bin:ref_hi_bin, :] ** 2, axis=0)
        ref_db     = 10.0 * np.log10(ref_energy + eps)

        # Smooth temporally — fast enough to follow attacks (~40ms at hop=1024/48kHz)
        ref_db = ndimage.gaussian_filter1d(
            ref_db.astype(np.float32), sigma=2.0, mode='nearest'
        ).astype(np.float64)

        # Scale: dB difference from median → linear multiplier
        median_db  = float(np.median(ref_db))
        delta_db   = ref_db - median_db
        gain_scale = np.clip(10.0 ** (delta_db / 20.0), gain_min, gain_max)

        # ─── 3. Per-frame Gaussian bell boost ──────────────────────
        mag_boosted = mag.copy()

        for f_idx in range(num_frames):
            # Bell center = seam + offset
            center = int(round(seam_f[f_idx])) + offset_bins
            center = int(np.clip(center, search_lo_bin, num_bins - 8))
            center_hz = center * bin_hz

            # σ in bins from Q:  BW = f_c / Q,  σ_hz = BW / 2.355
            sigma_bins = max(3.0, (center_hz / Q) / (2.355 * bin_hz))

            # Gaussian gain envelope (dB)
            g_db = peak_db * np.exp(-0.5 * ((bin_indices - center) / sigma_bins) ** 2)

            # Dynamic scaling + onset mask
            g_db *= gain_scale[f_idx]
            g_db *= onset_mask
            g_db = np.maximum(g_db, 0.0)  # boost only, never cut

            # Apply
            mag_boosted[:, f_idx] = mag[:, f_idx] * (10.0 ** (g_db / 20.0))

        # Reconstruct
        y = stft_engine.istft(mag_boosted * np.exp(1j * phase), length=len(x))

        if audio.ndim == 2:
            result[:, ch] = y
        else:
            result = y

    # Soft peak protection (allow some headroom for the 9 dB boost)
    orig_peak = float(np.max(np.abs(audio))) + eps
    res_peak  = float(np.max(np.abs(result))) + eps
    if res_peak > orig_peak * 1.5:
        result *= (orig_peak * 1.5 / res_peak)

    avg_seam = float(np.mean(seam_hz_report))
    print(f"    Dynamic seam bridge: detected seam ~{avg_seam:.0f} Hz, "
          f"boost center ~{avg_seam + offset_hz:.0f} Hz, "
          f"+{peak_db:.1f} dB Q={Q:.0f}, "
          f"energy range [{gain_min:.2f}x .. {gain_max:.2f}x]")

    return result


def _call_audiosr_build_model(build_model_fn, model_name: str, device_str: str = "cuda"):
    """
    Robust wrapper around audiosr.build_model across versions.

    Tries to call build_model with the right argument names/positions, then
    moves the returned model to the requested device when possible.
    """
    # Try to learn accepted parameters
    try:
        sig = inspect.signature(build_model_fn)
        params = list(sig.parameters.keys())
        param_set = set(params)
    except Exception:
        params = []
        param_set = set()

    attempts = []

    # Keyword attempts with common device arg names
    if param_set:
        for dev_key in ("device", "device_str", "dev", "cuda", "accelerator"):
            if dev_key in param_set:
                attempts.append({"model_name": model_name, dev_key: device_str})
                attempts.append({"name": model_name, dev_key: device_str})
        # If no device key, just pass model name
        attempts.append({"model_name": model_name})
        attempts.append({"name": model_name})

    # Fallback keyword styles (even if signature unknown)
    attempts.append({"model_name": model_name, "device": device_str})
    attempts.append({"model_name": model_name, "device_str": device_str})
    attempts.append({"name": model_name, "device": device_str})
    attempts.append({"name": model_name, "device_str": device_str})
    attempts.append({"model_name": model_name})
    attempts.append({"name": model_name})

    last_err = None

    # Try keyword calls
    for kw in attempts:
        try:
            # Filter kwargs if we know the signature
            if param_set:
                kw2 = {k: v for k, v in kw.items() if k in param_set}
            else:
                kw2 = dict(kw)
            model = build_model_fn(**kw2)
            # Move to device if possible
            try:
                if hasattr(model, "to"):
                    model = model.to(device_str)
                if hasattr(model, "eval"):
                    model.eval()
            except Exception:
                pass
            return model
        except Exception as e:
            last_err = e

    # Try positional calls
    for args in ((model_name, device_str), (model_name,), ()):
        try:
            model = build_model_fn(*args)
            try:
                if hasattr(model, "to"):
                    model = model.to(device_str)
                if hasattr(model, "eval"):
                    model.eval()
            except Exception:
                pass
            return model
        except Exception as e:
            last_err = e

    raise last_err


def _call_audiosr_super_resolution(super_resolution_fn, audiosr_model, inp: np.ndarray, **kwargs):
    """
    Robust wrapper around audiosr.pipeline.super_resolution across versions.

    - Sanitizes input/output to avoid NaN/Inf (librosa will crash otherwise)
    - Adapts step argument name if the installed audiosr doesn't accept 'steps'
      (common alternates: ddim_steps, num_steps, n_steps)
    - Retries with progressively smaller kwarg sets if needed.
    """
    # Sanitize only if this is a numeric buffer (no-op for file paths)
    inp = _ensure_finite_audio(inp, clamp=1.0)

    # Patch AudioSR lowpass to avoid SciPy ellip Wn==0/1 crashes
    try:
        _patch_audiosr_lowpass()
    except Exception:
        pass

    # Build kwargs adaptively based on signature if possible
    try:
        sig = inspect.signature(super_resolution_fn)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    def _filtered_kwargs(kws):
        if not params:
            return dict(kws)
        return {k: v for k, v in kws.items() if k in params}

    # Map "steps" to whatever the function supports
    if "steps" in kwargs and params and "steps" not in params:
        steps_val = kwargs.pop("steps")
        for alt in ("ddim_steps", "num_steps", "n_steps", "sampling_steps", "ddpm_steps"):
            if alt in params:
                kwargs[alt] = steps_val
                break

    # Common param name difference: guidance_scale vs guidance
    if "guidance_scale" in kwargs and params and "guidance_scale" not in params and "guidance" in params:
        kwargs["guidance"] = kwargs.pop("guidance_scale")

    # Try calling with progressively simpler kwargs
    attempts = []
    attempts.append(_filtered_kwargs(kwargs))
    attempts.append({k: v for k, v in attempts[0].items()
                     if k not in {"steps", "ddim_steps", "num_steps", "n_steps", "sampling_steps", "ddpm_steps"}})
    attempts.append({})

    last_err = None
    for kw in attempts:
        try:
            out = super_resolution_fn(audiosr_model, inp, **kw)
            out = _ensure_finite_audio(out, clamp=1.0)
            return out
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if 'digital filter critical frequencies' in msg or 'critical frequencies' in msg:
                try:
                    _patch_audiosr_lowpass()
                except Exception:
                    pass
            if ("not finite" in msg) or ("nan" in msg) or ("inf" in msg):
                inp = _ensure_finite_audio(inp, clamp=0.98)
            continue

    raise last_err


# =========================
# Multiband Crossfade (keep original below cutoff, AudioSR above)
# =========================

def _multiband_crossfade(original: np.ndarray, audiosr_out: np.ndarray,
                         crossover_hz: float, sr: int = 48000,
                         order: int = 8) -> np.ndarray:
    """Linkwitz-Riley crossover: original below crossover_hz, AudioSR above.

    This is the single most important post-processing step for AudioSR quality.
    AudioSR's vocoder subtly alters everything below the bandwidth boundary —
    bass, mids, transients, phase. By keeping the original signal below the
    crossover and only taking AudioSR's contribution above it, we preserve
    100% of the original tonal character while gaining the synthesised highs.

    Uses a Linkwitz-Riley topology (cascaded Butterworth²) for flat magnitude
    sum at the crossover point.

    DYNAMICS GATING: AudioSR is a diffusion model that generates roughly
    constant-level HF regardless of input dynamics.  We modulate the HF band
    by a fast envelope derived from the original signal so the synthesised
    highs breathe with the music — punchy on attacks, quiet during decays.
    """
    nyquist = sr / 2.0
    crossover_hz = float(np.clip(crossover_hz, 200.0, nyquist - 500.0))
    Wn = crossover_hz / nyquist
    # Clamp to valid filter range
    Wn = float(np.clip(Wn, 0.001, 0.999))

    try:
        sos_lo = signal.butter(order // 2, Wn, btype='low', output='sos')
        sos_hi = signal.butter(order // 2, Wn, btype='high', output='sos')
    except Exception:
        # Fallback to lower order if filter design fails
        sos_lo = signal.butter(2, Wn, btype='low', output='sos')
        sos_hi = signal.butter(2, Wn, btype='high', output='sos')

    # Ensure same length
    min_len = min(original.shape[0], audiosr_out.shape[0])
    orig = original[:min_len].copy()
    asr = audiosr_out[:min_len].copy()

    # Ensure 2D
    if orig.ndim == 1:
        orig = orig[:, np.newaxis]
    if asr.ndim == 1:
        asr = asr[:, np.newaxis]

    # ─── Dynamics envelope from original signal ───────────────────
    # Compute a fast broadband amplitude envelope with asymmetric attack/release
    # so HF tracks the original dynamics precisely.
    eps = 1e-12
    orig_mono = np.mean(np.abs(orig), axis=1)  # (min_len,)

    # Attack: ~1ms — short max_filter captures peaks instantly
    # Release: ~25ms — causal max_filter holds peaks for this duration then drops
    attack_win = max(3, int(sr * 0.001)) | 1  # must be odd
    release_win = max(3, int(sr * 0.025)) | 1

    try:
        # Step 1: fast peak tracker (attack) — catches transients
        env_attack = ndimage.maximum_filter1d(orig_mono, size=attack_win,
                                               mode='nearest')
        # Step 2: causal peak-hold (release) — origin shifts the window to
        # look backward only, so peaks hold for release_win samples then drop.
        # This gives fast attack (new peak → instant) + bounded release.
        env = ndimage.maximum_filter1d(env_attack, size=release_win,
                                        origin=release_win // 2, mode='nearest')
    except Exception:
        env = ndimage.maximum_filter1d(orig_mono, size=release_win, mode='nearest')

    # Normalize: peak of envelope = 1.0 (preserve absolute HF level at peaks)
    env_peak = np.max(env) + eps
    env_norm = env / env_peak

    # Apply a mild power curve so quiet parts drop a bit faster than linear
    # (prevents low-level "haze" while keeping natural dynamics)
    env_norm = env_norm ** 1.3

    # Floor: don't gate below -40 dB (allow minimal ambience through)
    env_floor = 0.01
    env_norm = np.maximum(env_norm, env_floor)

    result = np.zeros_like(orig)
    for ch in range(orig.shape[1]):
        lo = signal.sosfiltfilt(sos_lo, orig[:, ch]).astype(np.float64)
        hi = signal.sosfiltfilt(sos_hi, asr[:, ch]).astype(np.float64)
        # 23 kHz lowpass on the AudioSR HF band — removes ultrasonic vocoder
        # artifacts above the audible range (from Jarredou AudioSR fork)
        if sr >= 46000:
            try:
                sos_23k = signal.butter(2, 23000.0 / nyquist, btype='low', output='sos')
                hi = signal.sosfiltfilt(sos_23k, hi).astype(np.float64)
            except Exception:
                pass
        # ── Dynamics gate: modulate AudioSR HF by original envelope ──
        hi *= env_norm
        result[:, ch] = lo + hi

    return result


# =========================
# AudioSR Single Channel Helper (FIXED for stereo preservation)
# =========================

def _run_audiosr_single_channel(audio_mono: np.ndarray, sr: int, cfg: UltimateGPUConfig,
                                 audiosr_model, super_resolution, device: str) -> Tuple[np.ndarray, int]:
    """
    Run AudioSR on a single channel (mono) audio.
    Returns (output_mono, output_sr).
    
    This is the core AudioSR processing extracted for per-channel use.
    """
    # Pre-conditioning
    x = audio_mono.astype(np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]  # (L, 1)
    
    try:
        x = _lowpass_filtfilt(x, sr, cfg.PRE_LPF_HZ, order=4)
    except Exception:
        pass

    # Chunking
    chunk_sec = float(getattr(cfg, "AUDIOSR_CHUNK_SECONDS", 5.12))
    overlap_sec = float(getattr(cfg, "AUDIOSR_OVERLAP_SECONDS", 0.64))
    chunk_sec = max(1.0, min(chunk_sec, 10.24))
    overlap_sec = max(0.0, min(overlap_sec, chunk_sec * 0.49))

    chunk_len = int(chunk_sec * sr)
    overlap = int(overlap_sec * sr)
    hop = max(1, chunk_len - overlap)

    def _cos_fade(n: int) -> np.ndarray:
        if n <= 0:
            return np.ones((0,), dtype=np.float64)
        t = np.linspace(0.0, np.pi, n, dtype=np.float64)
        return 0.5 - 0.5 * np.cos(t)

    fade_in = _cos_fade(overlap)
    fade_out = fade_in[::-1].copy()

    use_fp16 = bool(getattr(cfg, "AUDIOSR_USE_FP16", True))

    n = x.shape[0]

    # ── AudioSR ALWAYS outputs at 48 kHz regardless of input sr ──
    output_sr = 48000
    sr_ratio = output_sr / float(sr) if sr != output_sr else 1.0

    # Overlap-add buffer must be in OUTPUT sample-rate space
    n_out = int(np.ceil(n * sr_ratio))
    y = np.zeros((n_out, 1), dtype=np.float64)
    wsum = np.zeros((n_out,), dtype=np.float64)

    # Fade / hop in output-rate space
    overlap_out = int(np.round(overlap * sr_ratio))
    hop_out = max(1, int(np.round(hop * sr_ratio)))

    fade_in_out = _cos_fade(overlap_out)
    fade_out_out = fade_in_out[::-1].copy()

    def _run_chunk_wav(chunk: np.ndarray, sr0: int, dev: str, steps: int) -> np.ndarray:
        import soundfile as _sf
        import tempfile as _tempfile
        with _tempfile.TemporaryDirectory() as td:
            in_wav = os.path.join(td, "in.wav")
            _sf.write(in_wav, chunk.astype(np.float32), sr0)
            out_any = _call_audiosr_super_resolution(
                super_resolution,
                audiosr_model,
                in_wav,
                seed=int(cfg.AUDIOSR_SEED),
                guidance_scale=float(cfg.AUDIOSR_GUIDANCE_SCALE),
                steps=int(steps),
                latent_tps=float(getattr(cfg, "AUDIOSR_LATENT_TPS", 12.8)),
                device_str=dev,
            )
            if hasattr(out_any, "detach"):
                out_np = out_any.detach().cpu().numpy()
            else:
                out_np = np.asarray(out_any)
            # Normalize to mono (L, 1)
            if out_np.ndim == 1:
                out_np = out_np[:, np.newaxis]
            elif out_np.ndim == 2:
                if out_np.shape[0] in (1, 2) and out_np.shape[1] > out_np.shape[0]:
                    out_np = out_np.T
                # Take mean if stereo
                if out_np.shape[1] > 1:
                    out_np = np.mean(out_np, axis=1, keepdims=True)
            return out_np.astype(np.float64)

    def _run_chunk_tensor(chunk: np.ndarray, sr0: int, dev: str, steps: int) -> np.ndarray:
        wf = torch.from_numpy(chunk.T.astype(np.float32))
        out_any = _call_audiosr_super_resolution(
            super_resolution,
            audiosr_model,
            wf,
            steps=int(steps),
            guidance_scale=float(cfg.AUDIOSR_GUIDANCE_SCALE),
            seed=int(cfg.AUDIOSR_SEED),
            device=dev,
        )
        if isinstance(out_any, (list, tuple)):
            out_any = out_any[0]
        out_np = out_any.detach().cpu().numpy().T
        if out_np.ndim == 1:
            out_np = out_np[:, np.newaxis]
        if out_np.shape[1] > 1:
            out_np = np.mean(out_np, axis=1, keepdims=True)
        return out_np.astype(np.float64)

    steps0 = int(cfg.AUDIOSR_DDIM_STEPS)
    steps_min = int(getattr(cfg, "AUDIOSR_DDIM_STEPS_MIN", 20))
    steps_min = max(5, min(steps_min, steps0))

    # Per-chunk loudness matching (like official AudioSR fork)
    _pyln_meter_in = None
    _pyln_meter_out = None
    try:
        import pyloudnorm as _pyln
        _pyln_meter_in = _pyln.Meter(sr)
        _pyln_meter_out = _pyln.Meter(48000)
    except ImportError:
        _ensure_pip_package("pyloudnorm", "pyloudnorm")
        try:
            import pyloudnorm as _pyln
            _pyln_meter_in = _pyln.Meter(sr)
            _pyln_meter_out = _pyln.Meter(48000)
        except Exception:
            _pyln = None
    except Exception:
        _pyln = None

    current_device = device

    pos = 0
    while pos < n:
        end = min(n, pos + chunk_len)
        chunk = x[pos:end, :]

        # pad last chunk for stability
        if (end - pos) < chunk_len:
            pad = chunk_len - (end - pos)
            chunk = np.pad(chunk, ((0, pad), (0, 0)), mode="reflect")

        steps = steps0
        out_chunk = None
        for _attempt in range(6):
            try:
                if current_device == "cuda" and use_fp16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        try:
                            out_chunk = _run_chunk_wav(_ensure_finite_audio(chunk, clamp=0.98), sr, current_device, steps)
                        except TypeError:
                            out_chunk = _run_chunk_tensor(chunk, sr, current_device, steps)
                else:
                    try:
                        out_chunk = _run_chunk_wav(_ensure_finite_audio(chunk, clamp=0.98), sr, current_device, steps)
                    except TypeError:
                        out_chunk = _run_chunk_tensor(chunk, sr, current_device, steps)
                break
            except RuntimeError as e:
                msg = str(e).lower()
                is_oom = ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)
                if current_device == "cuda" and is_oom:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    if steps > steps_min:
                        steps = max(steps_min, steps // 2)
                        continue
                    current_device = "cpu"
                    continue
                raise

        if out_chunk is None:
            return audio_mono, sr

        # ── Correct clip: the output is at 48 kHz, not input sr ──
        expected_out_len = int(np.round((end - pos) * sr_ratio))
        out_chunk = out_chunk[:expected_out_len, :]

        # Per-chunk loudness matching: match AudioSR output loudness to input
        if _pyln is not None and _pyln_meter_in is not None:
            try:
                in_1d = chunk[:end - pos, 0] if chunk.ndim == 2 else chunk[:end - pos]
                out_1d = out_chunk[:, 0] if out_chunk.ndim == 2 else out_chunk
                if len(in_1d) >= sr * 0.4 and len(out_1d) >= output_sr * 0.4:
                    loud_in = _pyln_meter_in.integrated_loudness(in_1d.astype(np.float64))
                    loud_out = _pyln_meter_out.integrated_loudness(out_1d.astype(np.float64))
                    if np.isfinite(loud_in) and np.isfinite(loud_out) and loud_in > -70 and loud_out > -70:
                        out_1d_matched = _pyln.normalize.loudness(out_1d.astype(np.float64), loud_out, loud_in)
                        if out_chunk.ndim == 2:
                            out_chunk[:, 0] = out_1d_matched[:out_chunk.shape[0]]
                        else:
                            out_chunk = out_1d_matched[:out_chunk.shape[0]]
            except Exception:
                pass  # Fall back to peak normalization if loudness matching fails

        # ── Overlap-add in output-rate coordinates ──
        pos_out = int(np.round(pos * sr_ratio))
        L_out = expected_out_len
        out_chunk = _normalize_chunk_shape_for_ola(out_chunk, target_len=L_out, target_ch=1)
        actual_len = out_chunk.shape[0]

        w = np.ones((actual_len,), dtype=np.float64)
        if pos > 0 and overlap_out > 0:
            nfi = min(overlap_out, actual_len)
            w[:nfi] *= fade_in_out[:nfi]
        if end < n and overlap_out > 0:
            nfo = min(overlap_out, actual_len)
            w[-nfo:] *= fade_out_out[-nfo:]

        out_end = min(n_out, pos_out + actual_len)
        write_len = out_end - pos_out
        y[pos_out:out_end, :] += out_chunk[:write_len] * w[:write_len, None]
        wsum[pos_out:out_end] += w[:write_len]

        if current_device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        pos += hop

    wsum = np.maximum(wsum, 1e-6)
    y = y / wsum[:, None]

    y = normalize_peak(y, min(0.95, float(np.max(np.abs(audio_mono))) + 1e-9))
    return y, output_sr


def _patch_msst_optional_imports(msst_repo: str):
    """Patch MSST source files so training-only imports can't crash inference.

    MSST's ``utils/settings.py`` (and sometimes ``train.py``) does
    ``import wandb`` at module level. wandb is a training logger — it's never
    called during inference — but the bare import kills the subprocess if the
    package isn't installed.

    This applies a minimal, non-destructive patch: wrap bare ``import X``
    lines in try/except so inference works regardless. The patch is
    idempotent (safe to call multiple times).
    """
    targets = {
        # file (relative to repo root) : list of module names to make optional
        os.path.join("utils", "settings.py"): ["wandb"],
        "train.py": ["wandb"],
    }

    for rel_path, modules in targets.items():
        fpath = os.path.join(msst_repo, rel_path)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                src = f.read()
        except Exception:
            continue

        modified = False
        for mod in modules:
            # Match  "import wandb"  that is NOT already inside a try block.
            # We look for the line at the start (or after whitespace) and wrap it.
            # Skip if we already patched (our sentinel comment is present).
            sentinel = f"# wizard5-patched-{mod}"
            if sentinel in src:
                continue

            # Replace bare  "import <mod>"  with a guarded version
            old = f"import {mod}"
            new = (
                f"{sentinel}\n"
                f"try:\n"
                f"    import {mod}\n"
                f"except ImportError:\n"
                f"    {mod} = None  # not needed for inference\n"
            )
            # Only replace top-level imports (not "from X import Y" or indented)
            lines = src.split("\n")
            new_lines = []
            for line in lines:
                stripped = line.strip()
                # Match exactly "import wandb" (possibly with trailing comment)
                if stripped == old or stripped.startswith(old + " ") or stripped.startswith(old + "#"):
                    # Check it's top-level (no leading whitespace or standard indent)
                    leading = len(line) - len(line.lstrip())
                    if leading == 0:
                        new_lines.append(new)
                        modified = True
                        continue
                new_lines.append(line)
            src = "\n".join(new_lines)

        if modified:
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(src)
            except Exception:
                pass

    # Note: The official MSST config_apollo.yaml already contains all required
    # keys (audio, training, inference, model). No source-level config patching
    # is needed. If config errors occur, delete models/apollo/config_apollo.yaml
    # to re-download the correct version.

    
    # Additional patch: fix _target_ keyword argument issue in base_model.py
    base_model_path = os.path.join(msst_repo, "models", "look2hear", "models", "base_model.py")
    if os.path.isfile(base_model_path):
        try:
            with open(base_model_path, "r", encoding="utf-8", errors="replace") as f:
                src = f.read()
            
            # Check if we've already patched this
            if "# wizard5-patched-target" not in src:
                modified = False
                
                # Find the apollo method and add filtering for _target_ and other special keys
                # We need to filter out Hydra/OmegaConf special keys before passing to model
                if "def apollo(" in src and "model = model_class(*args, **kwargs)" in src:
                    # Add a filter right before the model_class call
                    src = src.replace(
                        "model = model_class(*args, **kwargs)",
                        "# wizard5-patched-target\n        # Filter out Hydra/OmegaConf special keys\n        filtered_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('_')}\n        model = model_class(*args, **filtered_kwargs)"
                    )
                    modified = True
                
                # Write back if modified
                if modified:
                    with open(base_model_path, "w", encoding="utf-8") as f:
                        f.write(src)
                    print("      ✓ Patched _target_ keyword argument issue")
        except Exception as e:
            print(f"      ⚠️  Could not patch _target_ issue: {e}")


def _run_apollo_restore(audio: np.ndarray, sr: int, cfg: UltimateGPUConfig) -> np.ndarray:
    """
    Run Apollo lossy-audio restoration as a black-box first stage.

    Uses ZFTurbo's Music-Source-Separation-Training (MSST) framework, which
    natively supports Apollo via ``--model_type apollo``.

    Strategy (fully self-contained):
      1. Ensure MSST repo is cloned to  models/msst/  (once).
      2. Ensure Apollo config + checkpoint exist in  models/apollo/
         (uses local files; falls back to download if missing).
      3. Write input audio to a temp WAV.
      4. Run MSST ``inference.py`` as a subprocess.
      5. Read back the restored WAV output.
    """
    if not cfg.ENABLE_APOLLO:
        return audio

    if torch is None:
        print("  ⚠️  PyTorch unavailable; skipping Apollo restore.")
        return audio

    # ── Resolve paths ──────────────────────────────────────────────
    apollo_dir  = getattr(cfg, "APOLLO_DIR", os.path.join(_SCRIPT_DIR, "models", "apollo"))
    msst_dir    = getattr(cfg, "MSST_DIR",   os.path.join(_SCRIPT_DIR, "models", "msst"))
    cache_dir   = getattr(cfg, "CACHE_DIR",  os.path.join(_SCRIPT_DIR, "cache"))

    config_path = getattr(cfg, "APOLLO_CONFIG_PATH",
                          os.path.join(apollo_dir, "config_apollo.yaml"))
    ckpt_path   = getattr(cfg, "APOLLO_CHECKPOINT_PATH",
                          os.path.join(apollo_dir, "pytorch_model.bin"))

    for d in [apollo_dir, msst_dir, cache_dir]:
        os.makedirs(d, exist_ok=True)

    # ─── 1. Clone MSST repo (once) ─────────────────────────────────
    msst_repo = os.path.join(msst_dir, "Music-Source-Separation-Training")
    _msst_ready_marker = os.path.join(msst_repo, ".wizard5_ready")
    if not os.path.isdir(msst_repo):
        print("      Cloning MSST framework (first run only)...")
        try:
            subprocess.check_call(
                ["git", "clone", "--depth", "1",
                 "https://github.com/ZFTurbo/Music-Source-Separation-Training.git",
                 msst_repo],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"  ❌ Failed to clone MSST: {e}; skipping Apollo restore.")
            return audio

    # ─── 1b. Install deps + patch for inference (once) ─────────────
    if not os.path.isfile(_msst_ready_marker):
        print("      Installing MSST dependencies (first run only)...")
        # Install requirements.txt
        req_file = os.path.join(msst_repo, "requirements.txt")
        if os.path.isfile(req_file):
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-r", req_file,
                     "--quiet"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
            except Exception:
                print("      ⚠️  Some requirements.txt deps failed (non-fatal)")

        # Explicitly install deps that MSST imports but doesn't list in
        # requirements.txt (wandb is only used for training but imported
        # unconditionally in utils/settings.py)
        for pkg in ["wandb", "omegaconf", "ml_collections"]:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg,
                     "--quiet"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
            except Exception:
                pass  # will be handled by the patch below

        # Safety-net: patch utils/settings.py so `import wandb` can't crash
        # inference. wandb is only actually called during training, so a
        # try/except is perfectly safe.
        _patch_msst_optional_imports(msst_repo)

        # Mark as ready so we don't re-run on every invocation
        try:
            with open(_msst_ready_marker, "w") as f:
                f.write("ok\n")
        except Exception:
            pass
        print("      ✓ MSST framework ready")

    # ─── 2. Ensure config + checkpoint exist ───────────────────────
    def _download_if_missing(path: str, url_attr: str, label: str) -> bool:
        if os.path.isfile(path):
            return True
        url = getattr(cfg, url_attr, None)
        if not url:
            print(f"  ❌ {label} not found at {path} and no download URL configured.")
            return False
        print(f"      {label} not found locally — downloading...")
        try:
            subprocess.check_call(
                ["wget", "-q", "-O", path, url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            pass
        try:
            import urllib.request
            urllib.request.urlretrieve(url, path)
            return True
        except Exception as e2:
            print(f"  ❌ Download failed for {label}: {e2}")
            return False

    if not _download_if_missing(config_path, "APOLLO_CONFIG_URL", "Apollo config"):
        return audio
    if not _download_if_missing(ckpt_path, "APOLLO_CHECKPOINT_URL", "Apollo checkpoint (~1.5 GB)"):
        return audio

    # ─── 3. Write input to temp WAV ────────────────────────────────
    orig_audio = audio.copy()
    orig_sr = int(sr)
    orig_peak = float(np.max(np.abs(audio)))

    tmp_dir = tempfile.mkdtemp(prefix="apollo_", dir=cache_dir)
    input_wav = os.path.join(tmp_dir, "input.wav")
    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # ─── 3b. Create MSST-compatible config from user's config ──────
    # User configs may be in look2hear/Apollo native format (with keys like
    # 'datas', 'discriminator', 'optimizer_g') rather than MSST format
    # (with 'audio', 'training', 'inference'). MSST's inference.py expects
    # the latter, so we read the user's model params and generate a
    # temporary MSST-format config for the subprocess.
    msst_config_path = config_path  # default: use as-is
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}

        # Detect look2hear-format config (has 'datas' or '_target_' in model)
        is_look2hear = ("datas" in user_cfg
                        or (isinstance(user_cfg.get("model"), dict)
                            and "_target_" in user_cfg.get("model", {})))
        # Also detect missing MSST-required keys
        missing_msst_keys = not all(k in user_cfg for k in ("audio", "training", "inference"))

        if is_look2hear or missing_msst_keys:
            # Extract model params from user config
            m = user_cfg.get("model", {})
            model_sr = m.get("sr", 44100)
            model_win = m.get("win", 20)
            model_feature_dim = m.get("feature_dim", 256)
            model_layer = m.get("layer", 6)

            # Extract sample rate from datas if available
            d = user_cfg.get("datas", {})
            data_sr = d.get("sr", model_sr)

            msst_yaml = f"""\
audio:
  chunk_size: 132300
  num_channels: 2
  sample_rate: {data_sr}
  min_mean_abs: 0.0

model:
  sr: {model_sr}
  win: {model_win}
  feature_dim: {model_feature_dim}
  layer: {model_layer}

training:
  instruments: ['restored', 'addition']
  target_instrument: 'restored'
  batch_size: 2
  num_steps: 1000
  num_epochs: 1000
  optimizer: 'prodigy'
  lr: 1.0
  patience: 2
  reduce_factor: 0.95
  coarse_loss_clip: true
  grad_clip: 0
  q: 0.95
  use_amp: true

augmentations:
  enable: false

inference:
  batch_size: 4
  num_overlap: 4
"""
            msst_config_path = os.path.join(cache_dir, "config_apollo_msst.yaml")
            with open(msst_config_path, "w", encoding="utf-8") as f:
                f.write(msst_yaml)
            print(f"      ✓ Generated MSST-compatible config (sr={model_sr}, "
                  f"win={model_win}, dim={model_feature_dim}, layers={model_layer})")
    except Exception as e:
        print(f"      ⚠️  Config conversion failed ({e}); using original config")

    # Apollo expects 44100 Hz input
    apollo_sr = 44100
    if sr != apollo_sr:
        print(f"      Resampling {sr} → {apollo_sr} Hz for Apollo...")
        feed_audio = resample_gpu(audio, sr, apollo_sr)
    else:
        feed_audio = audio.copy()

    sf.write(input_wav, feed_audio.astype(np.float32), apollo_sr, subtype="FLOAT")

    # ─── 4. Run MSST inference.py as subprocess ────────────────────
    inference_script = os.path.join(msst_repo, "inference.py")
    if not os.path.isfile(inference_script):
        print("  ❌ MSST inference.py not found; skipping Apollo restore.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return audio

    model_type = getattr(cfg, "APOLLO_MODEL_TYPE", "apollo")

    cmd = [
        sys.executable, inference_script,
        "--model_type", model_type,
        "--config_path", msst_config_path,
        "--start_check_point", ckpt_path,
        "--input_folder", tmp_dir,
        "--store_dir", output_dir,
    ]

    if torch.cuda.is_available():
        cmd.extend(["--device_ids", "0"])

    print(f"      Running Apollo ({model_type}) restore...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=600,
            cwd=msst_repo,
        )
        if result.returncode != 0:
            stderr_snippet = (result.stderr or "")[-500:]
            print(f"  ⚠️  Apollo subprocess returned code {result.returncode}")
            if stderr_snippet:
                print(f"      Last stderr: ...{stderr_snippet}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return audio
        # Log subprocess output for debugging
        if result.stdout:
            for line in result.stdout.strip().split('\n')[-5:]:
                print(f"      [Apollo] {line}")
    except subprocess.TimeoutExpired:
        print("  ⚠️  Apollo timed out (>10 min); skipping.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return audio
    except Exception as e:
        print(f"  ⚠️  Apollo subprocess failed: {e}; skipping.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return audio

    # ─── 5. Read back the restored output ──────────────────────────
    output_files = sorted(glob.glob(os.path.join(output_dir, "**", "*.wav"), recursive=True))
    if not output_files:
        output_files = sorted(glob.glob(os.path.join(output_dir, "**", "*.flac"), recursive=True))
    if not output_files:
        # Debug: show what's actually in the output directory
        print("  ⚠️  Apollo produced no output files; skipping.")
        for root, dirs, files in os.walk(tmp_dir):
            for fn in files:
                print(f"      [debug] {os.path.join(root, fn)}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return audio

    restored_audio, restored_sr = sf.read(output_files[0], always_2d=True)
    restored_audio = restored_audio.astype(np.float64)
    print(f"      Apollo output: {restored_audio.shape[0]} samples, "
          f"{restored_audio.shape[1]} ch, {restored_sr} Hz")

    # ─── 6. Resample back to original SR if needed ─────────────────
    if int(restored_sr) != orig_sr:
        print(f"      Resampling Apollo output {restored_sr} → {orig_sr} Hz...")
        restored_audio = resample_gpu(restored_audio, int(restored_sr), orig_sr)

    # ─── 7. Match original channel count ───────────────────────────
    orig_ch = orig_audio.shape[1] if orig_audio.ndim == 2 else 1
    if restored_audio.ndim == 1:
        restored_audio = restored_audio[:, np.newaxis]
    if restored_audio.shape[1] < orig_ch:
        restored_audio = np.repeat(restored_audio, orig_ch, axis=1)
    elif restored_audio.shape[1] > orig_ch:
        restored_audio = restored_audio[:, :orig_ch]

    # ─── 8. Length-match and peak-match ────────────────────────────
    orig_len = orig_audio.shape[0]
    if restored_audio.shape[0] > orig_len:
        restored_audio = restored_audio[:orig_len]
    elif restored_audio.shape[0] < orig_len:
        pad = np.zeros((orig_len - restored_audio.shape[0], restored_audio.shape[1]),
                        dtype=np.float64)
        restored_audio = np.concatenate([restored_audio, pad], axis=0)

    restored_peak = float(np.max(np.abs(restored_audio)))
    if restored_peak > 1e-8:
        restored_audio = restored_audio * (min(orig_peak, 0.95) / restored_peak)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"      ✓ Apollo restore complete, peak: {np.max(np.abs(restored_audio)):.4f}")
    return restored_audio



def _demucs_nyquist_cleanup(audio: np.ndarray, sr: int) -> np.ndarray:
    """Remove Nyquist energy buildup ("laser beam") from Demucs stem recombination.

    Demucs operates at 44.1 kHz.  When stems are separated and recombined, phase
    mismatches between stems accumulate at the Nyquist frequency (sr/2 = 22050 Hz).
    At the native sample rate this is invisible (it IS the boundary), but once the
    audio is oversampled the artifact appears as a bright spectral line.

    Fix: apply a steep lowpass 500 Hz below Nyquist to cleanly remove the buildup
    before oversampling exposes it.  This costs nothing audible — real musical
    content at 21.5–22.05 kHz is negligible.
    """
    nyq = sr / 2.0
    # Cut 500 Hz below Nyquist (e.g. 21550 Hz at 44.1 kHz)
    cutoff = nyq - 500.0

    if cutoff <= 0 or cutoff >= nyq:
        return audio

    try:
        # 8th-order Butterworth = very steep (~48 dB/oct), phase-neutral via filtfilt
        sos = signal.butter(8, cutoff, btype='lowpass', fs=sr, output='sos')
        if audio.ndim == 2:
            for ch in range(audio.shape[1]):
                audio[:, ch] = signal.sosfiltfilt(sos, audio[:, ch])
        else:
            audio = signal.sosfiltfilt(sos, audio)
        print(f"    ✓ Nyquist cleanup: lowpass at {cutoff:.0f} Hz (kills {nyq:.0f} Hz artifact)")
    except Exception as e:
        print(f"    ⚠️  Nyquist cleanup failed: {e}")

    return audio


def _run_demucs_stem_fix(audio: np.ndarray, sr: int, cfg: UltimateGPUConfig) -> np.ndarray:
    """
    Stem-aware cleanup using Demucs v4 Hybrid Transformer models.

    Tries htdemucs_ft (fine-tuned, cleaner) first, falls back to htdemucs,
    then to torchaudio's bundled HDEMUCS_HIGH_MUSDB_PLUS as last resort.

    Uses the ``demucs`` package via torch.hub for htdemucs_ft/htdemucs
    (the torchaudio pipeline doesn't expose these v4 models).

    Runs chunked overlap-add to keep VRAM safe on 8 GB GPUs.
    """
    if not cfg.ENABLE_STEM_FIX:
        return audio

    if torch is None:
        print("  ⚠️  PyTorch unavailable; skipping stem fix.")
        return audio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Redirect torch-hub model cache → models/demucs/
    demucs_cache = getattr(cfg, "DEMUCS_DIR", os.path.join(_SCRIPT_DIR, "models", "demucs"))
    os.makedirs(demucs_cache, exist_ok=True)
    _prev_torch_home = os.environ.get("TORCH_HOME")
    os.environ["TORCH_HOME"] = demucs_cache

    # ── Try loading v4 Hybrid Transformer via demucs package ──────────
    model = None
    model_sr = 44100  # htdemucs models operate at 44.1 kHz
    labels = ["drums", "bass", "other", "vocals"]
    model_name = getattr(cfg, "DEMUCS_MODEL_NAME", "htdemucs_ft")

    # Attempt 1: demucs package (htdemucs_ft / htdemucs)
    _demucs_apply = None
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model as _demucs_apply
    except ImportError:
        # Try to install the demucs package automatically.
        # dora-search must be installed first (build dep for demucs).
        print("  ○ demucs package not installed; attempting pip install …")
        _install_ok = True
        for _pkg in ["hydra-core>=1.3", "dora-search", "demucs"]:
            try:
                r = subprocess.run(
                    [sys.executable, "-m", "pip", "install", _pkg,
                     "--quiet", "--disable-pip-version-check"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    text=True,
                )
                if r.returncode != 0:
                    print(f"    ○ pip install {_pkg} failed (rc={r.returncode})")
                    if r.stderr:
                        for _ln in r.stderr.strip().splitlines()[-3:]:
                            print(f"      {_ln}")
                    _install_ok = False
                    break
            except Exception as _pe:
                print(f"    ○ pip install {_pkg} error: {_pe}")
                _install_ok = False
                break

        if _install_ok:
            # Invalidate import caches so freshly-installed packages are visible
            import importlib
            importlib.invalidate_caches()
            try:
                from demucs.pretrained import get_model
                from demucs.apply import apply_model as _demucs_apply
                print("  ✓ demucs package installed successfully")
            except ImportError as ie:
                print(f"  ○ demucs installed but import failed: {ie}")
                get_model = None
        else:
            get_model = None

    if get_model is not None:
        for name in ([model_name] if model_name else []) + ["htdemucs_ft", "htdemucs"]:
            try:
                model = get_model(name)
                model.to(device).eval()
                model_sr = int(model.samplerate)
                labels = list(model.sources)
                print(f"  ✓ Loaded Demucs v4 model: {name} ({model_sr} Hz)")
                break
            except Exception as e:
                print(f"    ○ get_model('{name}') failed: {e}")
                model = None
                continue
    else:
        get_model = None  # ensure it's defined

    # Attempt 2: torch.hub (downloads repo → then use demucs.pretrained from it)
    if model is None:
        _demucs_apply = None
        try:
            # Load the repo via torch.hub so it gets downloaded & added to sys.path
            print("  ○ Trying torch.hub to download demucs repo …")
            torch.hub.load("facebookresearch/demucs", "demucs",
                           trust_repo=True)  # just triggers download
        except Exception:
            pass  # we don't care if this call itself fails

        # The hub-downloaded source still needs dora-search at runtime
        try:
            import dora  # noqa: F401
        except ImportError:
            try:
                r = subprocess.run(
                    [sys.executable, "-m", "pip", "install",
                     "hydra-core>=1.3", "dora-search",
                     "--quiet", "--disable-pip-version-check"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                )
                if r.returncode != 0:
                    print(f"    ○ pip install dora-search failed (rc={r.returncode})")
                    if r.stderr:
                        for _ln in r.stderr.strip().splitlines()[-3:]:
                            print(f"      {_ln}")
                else:
                    import importlib
                    importlib.invalidate_caches()
            except Exception:
                pass

        # Now try importing from the downloaded repo on sys.path
        try:
            from demucs.pretrained import get_model as _hub_get_model
            from demucs.apply import apply_model as _demucs_apply
            for name in [model_name, "htdemucs_ft", "htdemucs"]:
                try:
                    model = _hub_get_model(name)
                    model.to(device).eval()
                    model_sr = int(model.samplerate)
                    labels = list(model.sources)
                    print(f"  ✓ Loaded Demucs v4 via torch.hub repo: {name} ({model_sr} Hz)")
                    break
                except Exception as e:
                    print(f"    ○ torch.hub get_model('{name}') failed: {e}")
                    model = None
                    continue
        except ImportError as ie:
            print(f"    ○ Could not import demucs from torch.hub repo: {ie}")

    # Attempt 3: torchaudio bundled HDEMUCS (fallback)
    if model is None:
        _demucs_apply = None
        try:
            import torchaudio
            from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
            try:
                from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS as _PLUS
                bundle = _PLUS
            except Exception:
                bundle = HDEMUCS_HIGH_MUSDB
            model = bundle.get_model().to(device).eval()
            model_sr = int(getattr(bundle, "sample_rate", sr))
            for attr in ("get_source_labels", "get_labels", "get_source_names"):
                fn = getattr(bundle, attr, None)
                if callable(fn):
                    try:
                        labels = list(fn())
                        break
                    except Exception:
                        pass
            print(f"  ✓ Loaded torchaudio bundled HDemucs (fallback, {model_sr} Hz)")
        except Exception as e:
            print(f"  ⚠️  All Demucs backends failed; skipping stem fix: {e}")
            if _prev_torch_home is not None:
                os.environ["TORCH_HOME"] = _prev_torch_home
            else:
                os.environ.pop("TORCH_HOME", None)
            return audio

    # Restore env after model download
    if _prev_torch_home is not None:
        os.environ["TORCH_HOME"] = _prev_torch_home
    else:
        os.environ.pop("TORCH_HOME", None)

    # ── Prepare audio tensor ─────────────────────────────────────────
    x = audio.astype(np.float32)
    was_mono = (x.ndim == 1) or (x.ndim == 2 and x.shape[1] == 1)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    elif x.shape[1] == 1:
        x = np.concatenate([x, x], axis=1)  # (N,1) → (N,2)
    wf = torch.from_numpy(x.T)  # (C, T)  — guaranteed C==2

    import torchaudio as _ta_demucs
    if sr != model_sr:
        wf = _ta_demucs.functional.resample(wf, sr, model_sr)

    wf = wf.unsqueeze(0).to(device)  # (1, C, T)
    T = wf.shape[-1]

    # ── Run separation ───────────────────────────────────────────────
    # If demucs.apply.apply_model is available, use it (handles chunking
    # and overlap internally with shifts for better quality).
    if _demucs_apply is not None:
        try:
            with torch.no_grad():
                # apply_model returns (batch, sources, channels, time)
                # shifts=1 for speed, overlap=0.25 for quality
                est = _demucs_apply(model, wf, shifts=1, overlap=0.25,
                                    device=device, progress=False)
            out = est[0]  # (sources, C, T)
        except Exception as e:
            print(f"    ⚠️  demucs apply_model failed ({e}); falling back to manual chunking")
            _demucs_apply = None

    # Manual chunked overlap-add (fallback / torchaudio models)
    if _demucs_apply is None:
        chunk_sec = float(getattr(cfg, "DEMUCS_CHUNK_SECONDS", 20.0))
        overlap_sec = float(getattr(cfg, "DEMUCS_OVERLAP_SECONDS", 3.0))
        chunk_sec = max(5.0, min(chunk_sec, 60.0))
        overlap_sec = max(0.0, min(overlap_sec, chunk_sec * 0.49))

        chunk_len = int(chunk_sec * model_sr)
        overlap = int(overlap_sec * model_sr)
        hop = max(1, chunk_len - overlap)

        def _fade(n: int) -> torch.Tensor:
            if n <= 0:
                return torch.ones((0,), device=device)
            t = torch.linspace(0, math.pi, n, device=device)
            return 0.5 - 0.5 * torch.cos(t)

        fade_in = _fade(overlap)
        fade_out = torch.flip(fade_in, dims=[0])

        n_src = len(labels)
        out = torch.zeros((n_src, wf.shape[1], T), device=device, dtype=torch.float32)
        wsum = torch.zeros((T,), device=device, dtype=torch.float32)

        starts = list(range(0, max(1, T), hop))
        if starts and starts[-1] + chunk_len < T:
            starts.append(max(0, T - chunk_len))

        with torch.no_grad():
            for s in starts:
                e = min(T, s + chunk_len)
                seg = wf[:, :, s:e]
                pad = chunk_len - (e - s)
                if pad > 0:
                    seg = torch.nn.functional.pad(seg, (0, pad))

                est = model(seg)[0]  # (sources, C, chunk_len)
                est = est[:, :, : (e - s)]

                w = torch.ones((e - s,), device=device)
                if s > 0 and overlap > 0:
                    nfi = min(overlap, e - s)
                    w[:nfi] *= fade_in[:nfi]
                if e < T and overlap > 0:
                    nfo = min(overlap, e - s)
                    w[-nfo:] *= fade_out[-nfo:]

                out[:, :, s:e] += est * w[None, None, :]
                wsum[s:e] += w

        wsum = torch.clamp(wsum, min=1e-6)
        out = out / wsum[None, None, :]

    # ── Recombine stems with per-stem processing ─────────────────────
    stems = {lab: out[i].detach().cpu().numpy().T for i, lab in enumerate(labels)}

    drums = stems.get("drums")
    bass = stems.get("bass")
    other = stems.get("other")
    vocals = stems.get("vocals")

    if drums is None or bass is None or other is None or vocals is None:
        y = np.zeros((T, wf.shape[1]), dtype=np.float64)
        for _, v in stems.items():
            y += v.astype(np.float64)
    else:
        y = (drums + bass + other + vocals).astype(np.float64)
        try:
            y += 0.08 * _transient_enhance(drums.astype(np.float64), model_sr, strength=0.9)
        except Exception:
            pass
        try:
            voc_other = (vocals + other).astype(np.float64)
            voc_other_smooth = _deharsh_degrain(voc_other, model_sr, amount=0.8)
            # Subtract the harsh COMPONENT (original minus smoothed), not the smoothed signal
            y -= 0.06 * (voc_other - voc_other_smooth)
        except Exception:
            pass

    if sr != model_sr:
        y_t = torch.from_numpy(y.T.astype(np.float32))
        y_t = _ta_demucs.functional.resample(y_t, model_sr, sr)
        y = y_t.numpy().T.astype(np.float64)

    y = normalize_peak(y, min(0.95, float(np.max(np.abs(audio))) + 1e-9))
    if float(np.max(np.abs(y)) + 1e-12) > 0.995:
        y = y * (0.995 / (float(np.max(np.abs(y)) + 1e-12)))

    # Convert back to mono if input was mono
    if was_mono and y.ndim == 2 and y.shape[1] >= 2:
        y = np.mean(y, axis=1, keepdims=True)  # (N,2) → (N,1)

    return y

def process_file_gpu(
    input_path: str,
    output_dir: str,
    target_sr: Optional[int] = None,
    enable_hfr: bool = True,
    enable_dsre: bool = True,
    enable_declipping: bool = True,
    config: UltimateGPUConfig = None,
):
    """GPU-accelerated processing pipeline with proper gain staging.

    Pipeline (12 stages):
      1.  Apollo lossy-audio restoration
      2.  Demucs stem-aware cleanup (htdemucs_ft, pre-oversampling)
      3.  Audio analysis & parameter tuning
      4.  Oversampling (2× for cleaner DSP)
      5.  Pre-enhancement declipping
      6.  Resampling (optional explicit --sr target)
      7.  Mix polish (transient enhance + de-harsh)
      8.  HFPv2 MDCT harmonic extrapolation
      9.  MP-SENet phase restoration
     10.  Spectral flux correction (temporal smearing fix)
     11.  DSRE conservative detail recovery
     12.  Final processing (seam bridge, naturalization, limiter, output)
    """
    TOTAL_STAGES = 12
    config = config or UltimateGPUConfig()

    print(f"\n{'='*70}")
    print(f"GPU-ACCELERATED AUDIO ENHANCEMENT (STEREO PRESERVED)")
    print(f"Backend: {BACKEND.upper()}")
    print(f"Processing: {input_path}")
    print(f"{'='*70}")

    # ── Read input (universal format support) ────────────────────────
    try:
        audio, sr, input_bit_depth = read_audio_any_format(input_path)
        ref_audio = np.array(audio, copy=True)
        ref_sr = int(sr)
    except Exception as e:
        print(f"❌ Error reading: {e}", file=sys.stderr)
        return

    audio = audio.astype(np.float64)
    input_sr = int(sr)

    # ── Determine output sr/bit depth ────────────────────────────────
    # Default: match input, with minimum of 44100 Hz / 16-bit
    out_sr = int(config.OUTPUT_SAMPLE_RATE) if int(config.OUTPUT_SAMPLE_RATE) > 0 else input_sr
    out_sr = max(44100, out_sr)
    out_bits = int(config.OUTPUT_BIT_DEPTH) if int(config.OUTPUT_BIT_DEPTH) > 0 else input_bit_depth
    out_bits = max(16, out_bits)
    out_fmt = config.OUTPUT_FORMAT or 'wav'

    # Store original peak for reference
    original_peak = np.max(np.abs(audio))
    target_peak = min(original_peak, 0.95)

    print(f"📊 {audio.shape[0]} samples, {audio.shape[1]} ch, {sr} Hz, {input_bit_depth}-bit")
    print(f"   Duration: {audio.shape[0]/sr:.2f}s")
    print(f"   Original peak: {original_peak:.4f}")
    print(f"   Output target: {out_sr} Hz / {out_bits}-bit / {out_fmt}")

    # ─── [1] Apollo — lossy-audio restoration ────────────────────────
    print(f"\n[1/{TOTAL_STAGES}] Apollo Lossy-Audio Restoration")
    if config.ENABLE_APOLLO:
        _print_progress(1, TOTAL_STAGES, "Running Apollo restore (if available)")
        audio = _run_apollo_restore(audio, sr, config)
        audio = normalize_peak(audio, target_peak)
        print(f"  ✓ Apollo stage complete, peak: {np.max(np.abs(audio)):.4f}")
    else:
        print("  ⏭  Skipped (--no-apollo)")

    # ─── [2] Stem-aware Fix (Demucs htdemucs_ft) — PRE-OVERSAMPLING ──
    print(f"\n[2/{TOTAL_STAGES}] Stem-aware Fix (Demucs v4 Hybrid Transformer)")
    if config.ENABLE_STEM_FIX:
        _print_progress(2, TOTAL_STAGES, "Separating stems and fixing punch/roughness (if available)")
        audio = _run_demucs_stem_fix(audio, sr, config)
        # Kill Nyquist energy buildup before oversampling exposes it
        audio = _demucs_nyquist_cleanup(audio, sr)
        audio = normalize_peak(audio, target_peak)
        print(f"  ✓ Stem fix stage complete, peak: {np.max(np.abs(audio)):.4f}")
    else:
        print("  ⏭  Skipped (--no-stems)")

    # ─── [3] Analysis ────────────────────────────────────────────────
    print(f"\n[3/{TOTAL_STAGES}] Audio Analysis")
    analysis = ParameterTuner.analyze(audio, sr)
    config = ParameterTuner.suggest_parameters(analysis, config)
    print(f"  HF Energy: {analysis['hf_energy_ratio']*100:.2f}%")
    print(f"  Bandwidth: {analysis['effective_bandwidth']:.0f} Hz")

    # ─── [4] Oversampling (2× for cleaner DSP) ──────────────────────
    oversampled = False
    pre_oversample_sr = int(sr)
    oversample_factor = int(getattr(config, 'OVERSAMPLE_FACTOR', 2))
    if oversample_factor > 1:
        target_internal_sr = sr * oversample_factor
        print(f"\n[4/{TOTAL_STAGES}] Oversampling {sr} → {target_internal_sr} Hz (×{oversample_factor} for DSP chain)")
        audio = resample_gpu(audio, sr, target_internal_sr)
        ref_audio_os = resample_gpu(ref_audio, ref_sr, target_internal_sr)
        sr = target_internal_sr
        ref_sr = target_internal_sr
        ref_audio = ref_audio_os
        oversampled = True
        print(f"  ✓ Internal processing rate: {sr} Hz")
    else:
        print(f"\n[4/{TOTAL_STAGES}] Oversampling skipped (disabled)")

    # ─── [5] Pre-declipping ──────────────────────────────────────────
    print(f"\n[5/{TOTAL_STAGES}] Pre-Enhancement Declipping")
    is_clipped, clip_ratio, clip_details = ClippingDetector.detect_clipping(audio)

    if enable_declipping and is_clipped:
        print(f"  ⚠️  Clipping detected: {clip_ratio*100:.2f}%")
        declipper = GPUDeclipper(config)
        thr = (clip_details["positive_threshold"] + clip_details["negative_threshold"]) / 2.0
        audio = declipper.declip(audio, sr, clipping_threshold=thr)
        audio = normalize_peak(audio, target_peak)
        print(f"  ✓ Declipping complete, peak: {np.max(np.abs(audio)):.4f}")
    else:
        print("  ✓ No significant clipping")

    # ─── [6] Resampling (explicit --sr) ──────────────────────────────
    if target_sr and target_sr > 0 and sr != target_sr:
        print(f"\n[6/{TOTAL_STAGES}] Resampling {sr} -> {target_sr} Hz")
        audio = resample_gpu(audio, sr, target_sr)
        sr = target_sr
        print("  ✓ Resampling complete")
    else:
        print(f"\n[6/{TOTAL_STAGES}] No explicit resampling needed")

    # ─── [7] Mix Polish ──────────────────────────────────────────────
    print(f"\n[7/{TOTAL_STAGES}] Mix Polish (gentle de-harsh only)")
    _print_progress(7, TOTAL_STAGES, "Gentle de-harsh")
    if config.TRANSIENT_STRENGTH > 0.01:
        audio = _transient_enhance(audio, sr, strength=config.TRANSIENT_STRENGTH)
    audio = _deharsh_degrain(audio, sr, amount=config.ROUGHNESS_REDUCTION)
    audio = normalize_peak(audio, target_peak)
    print(f"  ✓ Mix polish complete, peak: {np.max(np.abs(audio)):.4f}")

    # ─── [8] HFP V2 — MDCT harmonic extrapolation ───────────────────
    if enable_hfr:
        print(f"\n[8/{TOTAL_STAGES}] HFP V2 (MDCT + cepstrum harmonic extrapolation)")
        pre_peak = np.max(np.abs(audio))
        hfp = HFPv2Restorer(config)
        cutoff = getattr(config, 'HFR_CUTOFF_HZ', 0) or 0
        audio = hfp.restore(audio, sr, lowpass_hz=(cutoff if cutoff > 0 else None))
        post_peak = np.max(np.abs(audio))
        print(f"  ✓ HFPv2 complete, peak: {pre_peak:.4f} -> {post_peak:.4f}")
        if post_peak > target_peak:
            audio = normalize_peak(audio, target_peak)
    else:
        print(f"\n[8/{TOTAL_STAGES}] HFP V2 skipped")

    # ─── [9] MP-SENet Phase Restoration ─────────────────────────────
    print(f"\n[9/{TOTAL_STAGES}] MP-SENet Phase Restoration")
    if getattr(config, 'ENABLE_MPSENET', True):
        _print_progress(9, TOTAL_STAGES, "Phase-aware restoration (if available)")
        audio = _run_mpsenet_phase_restore(audio, sr, config)
        audio = normalize_peak(audio, target_peak)
    else:
        print("  ⏭  Skipped (--no-mpsenet)")

    # ─── [10] Spectral Flux Correction ───────────────────────────────
    print(f"\n[10/{TOTAL_STAGES}] Spectral Flux Correction (temporal smearing fix)")

    if getattr(config, 'ENABLE_SPECTRAL_FLUX_CORRECTION', True):
        print("  Spectral flux correction...")
        audio = _spectral_flux_correction(
            audio, sr,
            strength=float(getattr(config, 'SPECTRAL_FLUX_STRENGTH', 0.65)),
            flux_threshold=float(getattr(config, 'SPECTRAL_FLUX_THRESHOLD', 0.30)),
        )
        audio = normalize_peak(audio, target_peak)
        print(f"    ✓ Spectral flux correction complete, peak: {np.max(np.abs(audio)):.4f}")
    else:
        print("  ⏭  Spectral flux correction skipped")

    # ─── [12] DSRE — conservative detail recovery ────────────────────
    if enable_dsre:
        print(f"\n[11/{TOTAL_STAGES}] GPU DSRE ({config.DSRE_M} stages, gentle)")
        pre_dsre_peak = np.max(np.abs(audio))
        audio = zansei_gpu(
            audio.T, sr,
            m=config.DSRE_M,
            decay=config.DSRE_DECAY,
            pre_hp=config.DSRE_PRE_HP,
            post_hp=config.DSRE_POST_HP,
            filter_order=config.DSRE_FILTER_ORDER,
            num_passes=config.NUM_ENHANCEMENT_PASSES,
        ).T
        post_dsre_peak = np.max(np.abs(audio))
        print(f"  ✓ DSRE complete, peak: {pre_dsre_peak:.4f} -> {post_dsre_peak:.4f}")
        if post_dsre_peak > target_peak:
            audio = normalize_peak(audio, target_peak)
    else:
        print(f"\n[11/{TOTAL_STAGES}] DSRE disabled")

    # ─── [12] Final Processing ───────────────────────────────────────
    print(f"\n[12/{TOTAL_STAGES}] Final Processing")

    # HF seam bridge
    if getattr(config, 'DYNAMIC_SEAM_BRIDGE', False) and getattr(config, 'DYNAMIC_SEAM_DB', 0) > 0.1:
        print(f"  Applying dynamic seam bridge (+{config.DYNAMIC_SEAM_DB:.1f} dB, Q={config.DYNAMIC_SEAM_Q:.0f}, energy-following)...")
        audio = _dynamic_seam_bridge(audio, sr, config)
        audio = normalize_peak(audio, target_peak)
    elif config.SEAM_BOOST_DB > 0.1:
        print(f"  Applying static seam boost (+{config.SEAM_BOOST_DB:.1f} dB, -{config.SEAM_BOOST_ROLLOFF:.1f} dB/oct)...")
        audio = _hf_seam_boost(audio, sr,
                               seam_hz=float(config.SEAM_BOOST_HZ),
                               peak_boost_db=config.SEAM_BOOST_DB,
                               rolloff_db_per_oct=config.SEAM_BOOST_ROLLOFF,
                               onset_width_hz=config.SEAM_BOOST_ONSET_WIDTH)
        audio = normalize_peak(audio, target_peak)

    # HF naturalization
    print(f"  Naturalizing HF spectrum (frequency-proportional)...")
    audio = _hf_naturalize(audio, sr,
                           transition_hz=config.NATURALIZE_TRANSITION_HZ,
                           diffusion_strength=config.NATURALIZE_DIFFUSION,
                           noise_floor_db=config.NATURALIZE_NOISE_FLOOR_DB,
                           phase_diffusion_amount=config.NATURALIZE_PHASE_DIFFUSION)

    # Ultrasonic cleanup
    audio = _ultrasonic_cleanup(audio, sr, cutoff_hz=20500.0)

    # Envelope matching from reference (prevent loud/quiet drift)
    audio = _preserve_envelope_down_only(audio, ref_audio, sr)

    # Gentle transient reinjection from reference
    base_l = os.path.basename(input_path).lower()
    if not any(t in base_l for t in ["no-transient", "no_transient", "notransient"]):
        audio = _transient_peak_match_upward_only(audio, ref_audio, sr,
                                                   strength=0.4, max_boost_db=3.0)

    # ── Downsample from oversampled rate to output rate ──────────────
    if oversampled or sr != out_sr:
        final_sr = out_sr
        if sr != final_sr:
            print(f"  Resampling {sr} → {final_sr} Hz (output)")
            audio = resample_gpu(audio, sr, final_sr)
            sr = final_sr

    # True-peak safety
    audio = _true_peak_limiter(audio, sr, target_dbfs=config.TRUEPEAK_TARGET_DBFS, oversample=4)
    audio = np.clip(audio, -0.99, 0.99).astype(np.float32)

    final_peak = np.max(np.abs(audio))
    print(f"  ✓ Final peak: {final_peak:.4f}")

    # ── Save (flexible format) ───────────────────────────────────────
    base = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base}_gpu_enhanced.{out_fmt}")

    try:
        actual_path = write_audio_any_format(output_path, audio, sr,
                                              bit_depth=out_bits,
                                              output_format=out_fmt)
        print(f"\n✅ Saved: {actual_path}")
        print(f"   Format: {out_fmt.upper()} / {out_bits}-bit / {sr} Hz")
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"❌ Save failed: {e}", file=sys.stderr)


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated AI Music Enhancement"
    )
    
    parser.add_argument("inputs", nargs="+", help="Input audio files")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: ./output/ next to script)")
    parser.add_argument("--sr", type=int, default=0, help="Target sample rate")
    
    # Stages on/off
    parser.add_argument("--no-apollo", action="store_true", help="Disable Apollo restore")
    parser.add_argument("--no-hfr", action="store_true", help="Disable Analog Vitality")
    parser.add_argument("--no-dsre", action="store_true", help="Disable DSRE")
    parser.add_argument("--no-declipping", action="store_true")
    parser.add_argument("--no-stems", action="store_true", help="Disable Demucs")
    parser.add_argument("--no-mpsenet", action="store_true", help="Disable MP-SENet phase restoration")
    parser.add_argument("--no-spectral-flux", action="store_true",
                        help="Disable spectral flux correction (transient sharpening)")

    # Oversampling (default: 2× for cleaner DSP)
    parser.add_argument("--no-oversample", action="store_true",
                        help="Disable 2× oversampling for DSP stages")
    parser.add_argument("--oversample-factor", type=int, default=2,
                        help="Oversample factor (default: 2)")

    # Output format (default: match input sr/bits, minimum 44.1kHz/16-bit)
    parser.add_argument("--output-format", type=str, default=None,
                        choices=["wav", "flac", "mp3"],
                        help="Output format (default: wav)")
    parser.add_argument("--output-bits", type=int, default=None,
                        choices=[16, 24, 32],
                        help="Output bit depth (default: match input, min 16)")
    parser.add_argument("--output-sr", type=int, default=None,
                        help="Output sample rate in Hz (default: match input, min 44100)")

    # Spectral flux tuning
    parser.add_argument("--spectral-flux-strength", type=float, default=None,
                        help="Spectral flux correction strength 0..1 (default 0.65)")

    # Apollo
    parser.add_argument("--apollo-checkpoint", type=str, default=None,
                        help="Override Apollo checkpoint path (local file or URL)")
    parser.add_argument("--apollo-config", type=str, default=None,
                        help="Override Apollo config path (local file or URL)")

    # Demucs model selection
    parser.add_argument("--demucs-model", type=str, default=None,
                        choices=["htdemucs_ft", "htdemucs", "htdemucs_6s"],
                        help="Demucs model (default: htdemucs_ft)")

    # Vitality tuning (all have sane defaults in config)
    parser.add_argument("--saturation", type=float, default=None,
                        help="Saturation drive 0..1 (default 0.15)")
    parser.add_argument("--shelf-db", type=float, default=None,
                        help="HF shelf boost in dB (default 6)")
    parser.add_argument("--warmth", type=float, default=None,
                        help="HF warmth/rolloff 0..1 (default 0.3)")
    parser.add_argument("--vitality", type=float, default=None,
                        help="Overall vitality amount 0..1 (default 0.7)")
    parser.add_argument("--smooth", type=float, default=None,
                        help="Spectral smoothing strength 0..1 (default 0.5)")
    parser.add_argument("--hfr-cutoff", type=int, default=0,
                        help="Override detected cutoff in Hz (0 = auto)")

    # General
    parser.add_argument("--transient", type=float, default=None,
                        help="Transient strength 0..1 (default 0.10 — very gentle)")
    parser.add_argument("--roughness", type=float, default=None)
    parser.add_argument("--truepeak", type=float, default=-1.0)
    parser.add_argument("--gl-iterations", type=int, default=150)
    parser.add_argument("--declip-iterations", type=int, default=500)
    parser.add_argument("--dsre-m", type=int, default=None)

    # HF Naturalization
    parser.add_argument("--nat-transition", type=float, default=None,
                        help="Naturalize transition freq in Hz (default 11000)")
    parser.add_argument("--nat-diffusion", type=float, default=None,
                        help="Naturalize diffusion strength 0..2 (default 1.0)")
    parser.add_argument("--nat-noise", type=float, default=None,
                        help="Naturalize noise floor in dB (default -45)")
    parser.add_argument("--no-naturalize", action="store_true",
                        help="Disable HF naturalization entirely")

    # HF Seam Boost
    parser.add_argument("--seam-db", type=float, default=None,
                        help="Seam boost peak in dB (default 8)")
    parser.add_argument("--seam-hz", type=float, default=None,
                        help="Override seam frequency in Hz (default auto-detect)")
    parser.add_argument("--seam-rolloff", type=float, default=None,
                        help="Seam boost rolloff in dB/oct (default 6)")
    parser.add_argument("--no-seam-boost", action="store_true",
                        help="Disable seam boost entirely")

    # Dynamic Seam Bridge
    parser.add_argument("--no-dynamic-seam", action="store_true",
                        help="Disable dynamic seam bridge (falls back to static)")
    parser.add_argument("--dyn-seam-db", type=float, default=None,
                        help="Dynamic seam bridge peak dB (default 9)")
    parser.add_argument("--dyn-seam-q", type=float, default=None,
                        help="Dynamic seam bridge Q (default 9)")
    parser.add_argument("--dyn-seam-offset", type=float, default=None,
                        help="Hz above detected seam to place boost (default 600)")
    
    args = parser.parse_args()
    
    config = UltimateGPUConfig()
    config.GL_ITERATIONS = args.gl_iterations
    config.DECLIP_ITERATIONS = args.declip_iterations
    config.HFR_CUTOFF_HZ = int(args.hfr_cutoff)
    config.TRUEPEAK_TARGET_DBFS = float(args.truepeak)

    # Apollo
    config.ENABLE_APOLLO = not args.no_apollo
    if args.apollo_checkpoint is not None:
        # If it's a local path, use directly; otherwise treat as URL
        if os.path.isfile(args.apollo_checkpoint):
            config.APOLLO_CHECKPOINT_PATH = os.path.abspath(args.apollo_checkpoint)
        else:
            config.APOLLO_CHECKPOINT_URL = args.apollo_checkpoint
    if args.apollo_config is not None:
        if os.path.isfile(args.apollo_config):
            config.APOLLO_CONFIG_PATH = os.path.abspath(args.apollo_config)
        else:
            config.APOLLO_CONFIG_URL = args.apollo_config

    config.ENABLE_STEM_FIX = not args.no_stems
    config.ENABLE_MPSENET = not args.no_mpsenet
    config.ENABLE_SPECTRAL_FLUX_CORRECTION = not args.no_spectral_flux

    # Oversampling
    if args.no_oversample:
        config.OVERSAMPLE_FACTOR = 1
    else:
        config.OVERSAMPLE_FACTOR = int(args.oversample_factor)

    # Output format / bit depth / sample rate
    if args.output_format is not None:
        config.OUTPUT_FORMAT = args.output_format
    if args.output_bits is not None:
        config.OUTPUT_BIT_DEPTH = int(args.output_bits)
    if args.output_sr is not None:
        config.OUTPUT_SAMPLE_RATE = int(args.output_sr)

    # Spectral flux tuning
    if args.spectral_flux_strength is not None:
        config.SPECTRAL_FLUX_STRENGTH = float(args.spectral_flux_strength)
    # Demucs model selection
    if args.demucs_model is not None:
        config.DEMUCS_MODEL_NAME = args.demucs_model

    # Override vitality params if specified
    if args.saturation is not None:
        config.VITALITY_SATURATION_DRIVE = float(args.saturation)
    if args.shelf_db is not None:
        config.VITALITY_SHELF_GAIN_DB = float(args.shelf_db)
    if args.warmth is not None:
        config.VITALITY_WARMTH = float(args.warmth)
    if args.vitality is not None:
        config.VITALITY_AMOUNT = float(args.vitality)
    if args.smooth is not None:
        config.VITALITY_SMOOTH_STRENGTH = float(args.smooth)
    if args.transient is not None:
        config.TRANSIENT_STRENGTH = float(args.transient)
    if args.roughness is not None:
        config.ROUGHNESS_REDUCTION = float(args.roughness)
    if args.dsre_m is not None:
        config.DSRE_M = int(args.dsre_m)
    if args.nat_transition is not None:
        config.NATURALIZE_TRANSITION_HZ = float(args.nat_transition)
    if args.nat_diffusion is not None:
        config.NATURALIZE_DIFFUSION = float(args.nat_diffusion)
    if args.nat_noise is not None:
        config.NATURALIZE_NOISE_FLOOR_DB = float(args.nat_noise)
    if args.no_naturalize:
        config.NATURALIZE_DIFFUSION = 0.0
    if args.seam_db is not None:
        config.SEAM_BOOST_DB = float(args.seam_db)
    if args.seam_hz is not None:
        config.SEAM_BOOST_HZ = float(args.seam_hz)
    if args.seam_rolloff is not None:
        config.SEAM_BOOST_ROLLOFF = float(args.seam_rolloff)
    if args.no_seam_boost:
        config.SEAM_BOOST_DB = 0.0
        config.DYNAMIC_SEAM_DB = 0.0
    if args.no_dynamic_seam:
        config.DYNAMIC_SEAM_BRIDGE = False
    if args.dyn_seam_db is not None:
        config.DYNAMIC_SEAM_DB = float(args.dyn_seam_db)
    if args.dyn_seam_q is not None:
        config.DYNAMIC_SEAM_Q = float(args.dyn_seam_q)
    if args.dyn_seam_offset is not None:
        config.DYNAMIC_SEAM_OFFSET_HZ = float(args.dyn_seam_offset)
    
    target_sr = args.sr if args.sr > 0 else None

    # Default output dir: ./output/ next to the script
    output_dir = args.output_dir or config.OUTPUT_DIR

    # Ensure the organized directory structure exists
    for d in [config.MODELS_DIR, config.APOLLO_DIR,
              config.DEMUCS_DIR, config.MSST_DIR,
              getattr(config, 'MPSENET_DIR', ''),
              config.CACHE_DIR, output_dir]:
        if d:
            os.makedirs(d, exist_ok=True)

    print(f"📂 Directory layout:")
    print(f"   Models:  {config.MODELS_DIR}")
    print(f"   Cache:   {config.CACHE_DIR}")
    print(f"   Output:  {output_dir}")
    print()
    
    for input_path in args.inputs:
        if not os.path.isfile(input_path):
            print(f"❌ Skipping: {input_path}")
            continue
        
        process_file_gpu(
            input_path=input_path,
            output_dir=output_dir,
            target_sr=target_sr,
            enable_hfr=not args.no_hfr,
            enable_dsre=not args.no_dsre,
            enable_declipping=not args.no_declipping,
            config=config,
        )


if __name__ == "__main__":
    main()
