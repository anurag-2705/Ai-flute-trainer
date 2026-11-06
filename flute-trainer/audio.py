"""Audio sources: live microphone, or a WAV file played back in real time (for testing)."""
import time
import wave
from collections import deque

import numpy as np

import config


class MicSource:
    def __init__(self, sr=config.SAMPLE_RATE, size=config.FRAME_SIZE, device=None):
        import sounddevice as sd
        self.sr, self.size = sr, size
        self._buf = deque(maxlen=size)
        self._stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                      device=device, callback=self._cb)
        self._stream.start()

    def _cb(self, indata, frames, t, status):
        self._buf.extend(indata[:, 0])

    def latest(self):
        """Most recent frame, or None until enough audio has arrived."""
        if len(self._buf) < self.size:
            return None
        return np.array(self._buf, dtype=np.float32)

    def close(self):
        self._stream.stop()
        self._stream.close()


class FileSource:
    """Plays a mono/stereo 16-bit WAV as if it were a live microphone."""

    def __init__(self, path, size=config.FRAME_SIZE):
        with wave.open(str(path), "rb") as wf:
            self.sr = wf.getframerate()
            ch = wf.getnchannels()
            raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        self.data = (raw.reshape(-1, ch).mean(axis=1) / 32768.0).astype(np.float32)
        self.size = size
        self.t0 = time.time()

    def latest(self):
        pos = int((time.time() - self.t0) * self.sr)
        if pos < self.size:
            return None
        if pos > len(self.data):
            return None
        return self.data[pos - self.size:pos]

    @property
    def finished(self):
        return (time.time() - self.t0) * self.sr > len(self.data)

    def close(self):
        pass
