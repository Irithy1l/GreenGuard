import argparse
import json
import math
import random
import time
from datetime import datetime
from urllib import error, parse, request


SOIL_DRY = 800
SOIL_WET = 350
LIGHT_DARK = 200
LIGHT_BRIGHT = 900
LIGHT_DETECT_THRESHOLD = 650


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def pct_to_soil_raw(soil_percent):
    # Inverse of Arduino mapping:
    # soilPercent = map(soilRaw, SOIL_DRY, SOIL_WET, 0, 100)
    raw = SOIL_DRY + (soil_percent / 100.0) * (SOIL_WET - SOIL_DRY)
    return int(round(raw))


def percent_from_raw(raw, lo, hi):
    if hi == lo:
        return 0
    pct = (raw - lo) * 100.0 / (hi - lo)
    return int(round(clamp(pct, 0, 100)))


class FakeArduino:
    def __init__(self, base_url, interval_sec, username=None, user_id=None, seed=None):
        self.base_url = base_url.rstrip("/")
        self.interval_sec = interval_sec
        self.username = username
        self.user_id = user_id
        self.rng = random.Random(seed)
        self.ctrl_pin_high = False  # HIGH => irrigating YES
        self.tick = 0

        # Internal simulated state
        self.soil_percent_state = self.rng.uniform(35.0, 65.0)
        self.temp_state = self.rng.uniform(20.0, 27.0)
        self.humidity_state = self.rng.uniform(35.0, 65.0)

    def _url(self, path):
        url = f"{self.base_url}{path}"
        q = {}
        if self.username:
            q["username"] = self.username
        elif self.user_id:
            q["user_id"] = str(self.user_id)
        if q and path.startswith("/ingest"):
            url += "?" + parse.urlencode(q)
        return url

    def _http_get_json(self, path, timeout=2.5):
        req = request.Request(self._url(path), method="GET")
        with request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            return json.loads(data) if data else {}

    def _http_post_json(self, path, payload, timeout=2.5):
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self._url(path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            return json.loads(data) if data else {}

    def get_command(self):
        try:
            res = self._http_get_json("/cmd")
            cmd = str(res.get("cmd", "")).upper().strip()
            return cmd if cmd in ("START", "STOP") else ""
        except Exception as exc:
            print(f"[WARN] Failed GET /cmd: {exc}")
            return ""

    def apply_command(self, cmd):
        if cmd == "START":
            self.ctrl_pin_high = True
            print("CMD=START -> CTRL_PIN HIGH")
        elif cmd == "STOP":
            self.ctrl_pin_high = False
            print("CMD=STOP -> CTRL_PIN LOW")

    def is_irrigating(self):
        return self.ctrl_pin_high

    def build_sensor_payload(self):
        irrigating = self.is_irrigating()

        # Soil moisture dynamics
        if irrigating:
            self.soil_percent_state += self.rng.uniform(2.5, 6.0)
        else:
            self.soil_percent_state -= self.rng.uniform(0.4, 1.4)
        self.soil_percent_state = clamp(self.soil_percent_state, 0.0, 100.0)

        soil_raw = pct_to_soil_raw(self.soil_percent_state)
        soil_percent = percent_from_raw(soil_raw, SOIL_DRY, SOIL_WET)

        # Light: daily sinusoidal pattern + noise
        now = datetime.now()
        hour = now.hour + now.minute / 60.0
        daylight = max(0.0, math.sin((hour - 6.0) / 12.0 * math.pi))
        light_base = LIGHT_DARK + daylight * (LIGHT_BRIGHT - LIGHT_DARK)
        light_raw = int(round(clamp(light_base + self.rng.uniform(-60, 60), 0, 1023)))
        light_percent = percent_from_raw(light_raw, LIGHT_DARK, LIGHT_BRIGHT)
        light_detected = "YES" if light_raw >= LIGHT_DETECT_THRESHOLD else "NO"

        # Temp and humidity drift
        temp_target = 19.0 + daylight * 10.0
        self.temp_state += (temp_target - self.temp_state) * 0.2 + self.rng.uniform(-0.4, 0.4)
        self.temp_state = clamp(self.temp_state, 10.0, 40.0)

        hum_target = 70.0 - daylight * 25.0 + (4.0 if irrigating else 0.0)
        self.humidity_state += (hum_target - self.humidity_state) * 0.15 + self.rng.uniform(-1.8, 1.8)
        self.humidity_state = clamp(self.humidity_state, 15.0, 95.0)

        return {
            "temp_c": round(self.temp_state, 2),
            "humidity": round(self.humidity_state, 2),
            "soil_raw": int(soil_raw),
            "soil_percent": int(soil_percent),
            "light_raw": int(light_raw),
            "light_percent": int(light_percent),
            "light_detected": light_detected,
            "irrigating": "YES" if irrigating else "NO",
        }

    def post_ingest(self, payload):
        try:
            res = self._http_post_json("/ingest", payload)
            return True, res
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            print(f"[WARN] POST /ingest HTTP {exc.code}: {detail}")
            return False, {}
        except Exception as exc:
            print(f"[WARN] Failed POST /ingest: {exc}")
            return False, {}

    def run(self, once=False):
        print("Fake Arduino started.")
        print(f"Server: {self.base_url}")
        print(f"Interval: {self.interval_sec:.2f}s")
        if self.username:
            print(f"Ingest as username={self.username}")
        elif self.user_id:
            print(f"Ingest as user_id={self.user_id}")
        else:
            print("Ingest with default server user resolution.")
        print("-" * 36)

        while True:
            cmd = self.get_command()
            if cmd:
                self.apply_command(cmd)

            payload = self.build_sensor_payload()
            tstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(
                f"[{tstamp}] "
                f"T={payload['temp_c']}C  "
                f"H={payload['humidity']}%  "
                f"Soil={payload['soil_raw']} ({payload['soil_percent']}%)  "
                f"Light={payload['light_raw']} ({payload['light_percent']}%)  "
                f"Detected={payload['light_detected']}  "
                f"Irrigating={payload['irrigating']}  "
                f"CMD={cmd or '-'}"
            )

            ok, _ = self.post_ingest(payload)
            if ok:
                print("Uploaded to server OK")
            else:
                print("Upload failed")
            print("-" * 36)

            if once:
                break
            time.sleep(self.interval_sec)


def parse_args():
    p = argparse.ArgumentParser(description="Fake Arduino sensor uploader.")
    p.add_argument("--server", default="http://127.0.0.1:5000", help="Flask server base URL.")
    p.add_argument("--interval", type=float, default=2.0, help="Loop interval seconds.")
    p.add_argument("--username", default="", help="Username query for /ingest.")
    p.add_argument("--user-id", type=int, default=0, help="User ID query for /ingest.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--once", action="store_true", help="Run one loop only.")
    return p.parse_args()


def main():
    args = parse_args()
    fa = FakeArduino(
        base_url=args.server,
        interval_sec=args.interval,
        username=args.username or None,
        user_id=args.user_id if args.user_id > 0 else None,
        seed=args.seed,
    )
    fa.run(once=args.once)


if __name__ == "__main__":
    main()
