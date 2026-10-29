"""Pembacaan sensor Raspberry Pi 4: DHT22, BH1750, INMP441.

Wiring:
  DHT22   : DATA -> GPIO4 (pin 7), VCC 3.3V, GND
  BH1750  : SDA -> GPIO2 (pin 3), SCL -> GPIO3 (pin 5), VCC 3.3V, GND, ADDR -> GND (0x23)
  INMP441 : SCK -> GPIO18 (pin 12), WS -> GPIO19 (pin 35), SD -> GPIO20 (pin 38),
            L/R -> GND, VDD 3.3V, GND
Set SENSOR_MOCK=1 untuk data palsu saat development di PC.
"""
import collections
import math
import os
import random
import threading
import time

import numpy as np

MOCK = os.environ.get("SENSOR_MOCK") == "1"


class SensorHub:
    def __init__(self, interval=2.0):
        self.interval = interval
        self.data = {"suhu": None, "kelembapan": None, "cahaya": None, "suara": None}
        self._lock = threading.Lock()
        self.history = collections.deque(maxlen=300)  # ~10 menit @ 2 dtk
        self._db = None
        self._dht = self._bus = None

        if not MOCK:
            self._init_dht()
            self._init_bh1750()
            self._init_mic()
        threading.Thread(target=self._loop, daemon=True).start()

    # ---------- init ----------
    def _init_dht(self):
        try:
            import adafruit_dht
            import board
            self._dht = adafruit_dht.DHT22(board.D4)
        except Exception as e:
            print("[sensor] DHT22 gagal:", e)

    def _init_bh1750(self):
        try:
            from smbus2 import SMBus
            self._bus = SMBus(1)
        except Exception as e:
            print("[sensor] BH1750 gagal:", e)

    def _init_mic(self):
        try:
            import sounddevice as sd
            dev = os.environ.get("MIC_DEVICE")  # nama/index, opsional
            self._stream = sd.InputStream(
                device=int(dev) if dev and dev.isdigit() else dev,
                samplerate=48000, channels=2, dtype="int32",
                blocksize=4800, callback=self._mic_cb)
            self._stream.start()
        except Exception as e:
            print("[sensor] INMP441 gagal:", e)

    def _mic_cb(self, indata, frames, t, status):
        x = indata[:, 0].astype(np.float64) / 2**31    # kanal kiri (L/R -> GND)
        x -= x.mean()
        rms = math.sqrt(float(np.mean(x ** 2))) + 1e-12
        dbfs = 20 * math.log10(rms)
        # INMP441: sensitivitas -26 dBFS @ 94 dB SPL -> SPL ~ dBFS + 120 (perkiraan)
        db = max(0.0, dbfs + 120)
        self._db = db if self._db is None else 0.7 * self._db + 0.3 * db

    # ---------- baca ----------
    def _read_bh1750(self):
        self._bus.write_byte(0x23, 0x20)       # one-time high-res mode
        time.sleep(0.18)
        d = self._bus.read_i2c_block_data(0x23, 0x00, 2)
        return ((d[0] << 8) | d[1]) / 1.2

    def _loop(self):
        while True:
            new = {}
            if MOCK:
                new = {"suhu": 26 + random.random(), "kelembapan": 58 + random.random() * 2,
                       "cahaya": 420 + random.random() * 20, "suara": 40 + random.random() * 5}
            else:
                if self._dht:
                    try:
                        new["suhu"] = self._dht.temperature
                        new["kelembapan"] = self._dht.humidity
                    except RuntimeError:
                        pass            # DHT22 sering gagal sesaat, pakai nilai lama
                    except Exception as e:
                        print("[sensor] DHT22:", e)
                if self._bus:
                    try:
                        new["cahaya"] = self._read_bh1750()
                    except Exception as e:
                        print("[sensor] BH1750:", e)
                if self._db is not None:
                    new["suara"] = self._db
            with self._lock:
                self.data.update({k: v for k, v in new.items() if v is not None})
                self.history.append({"t": int(time.time() * 1000), **self.data})
            time.sleep(self.interval)

    def get_history(self):
        with self._lock:
            return list(self.history)

    def get(self):
        with self._lock:
            return dict(self.data)
