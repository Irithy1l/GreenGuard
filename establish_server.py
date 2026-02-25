from flask import Flask, request, jsonify, Response
from datetime import datetime
import time
import threading
import sys
import logging

# Optional: hide request spam
# If you want to show where server is (in terms of IP)
# Flip this to False
logging.getLogger('werkzeug').disabled = True

app = Flask(__name__)

latest = {
    "time": None,
    "temp_c": None,
    "humidity": None,
    "soil_raw": None,
    "soil_percent": None,
    "light_raw": None,
    "light_percent": None,
    "light_detected": None,
    "irrigating": None,   # <-- NEW
}

# Command state set by webpage; Arduino polls it
command_state = {"cmd": "STOP"}  # default

# Watchdog: warn if uploads stop
last_ingest_ts = None
TIMEOUT_SEC = 6.0
CHECK_EVERY_SEC = 1.0


@app.post("/ingest")
def ingest():
    global latest, last_ingest_ts
    data = request.get_json(force=True, silent=True) or {}

    latest.update(data)
    latest["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_ingest_ts = time.time()

    # Logs in PyCharm
    print(
        f"\n[{latest['time']}] "
        f"T={latest.get('temp_c')}°C  "
        f"H={latest.get('humidity')}%  "
        f"Soil={latest.get('soil_raw')} ({latest.get('soil_percent')}%)  "
        f"Light={latest.get('light_raw')} ({latest.get('light_percent')}%)  "
        f"Detected={latest.get('light_detected')}  "
        f"Irrigating={latest.get('irrigating')}  "
        f"CMD={command_state.get('cmd')}"
    )
    sys.stdout.flush()

    return jsonify({"ok": True})


@app.get("/latest")
def get_latest():
    return jsonify(latest)


@app.get("/cmd")
def get_cmd():
    return jsonify(command_state)


@app.post("/cmd")
def set_cmd():
    data = request.get_json(force=True, silent=True) or {}
    cmd = (data.get("cmd") or "").upper().strip()
    if cmd not in ("START", "STOP"):
        return jsonify({"ok": False, "error": "cmd must be START or STOP"}), 400

    command_state["cmd"] = cmd
    print(f"\n[CMD] Set to {cmd}\n")
    sys.stdout.flush()
    return jsonify({"ok": True, "cmd": cmd})


@app.get("/")
def index():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>UNO Sensor Dashboard</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 24px; }
        .card { padding: 16px; border: 1px solid #ddd; border-radius: 16px; max-width: 640px; }
        .row { margin: 8px 0; display: flex; gap: 12px; align-items: center; }
        .k { width: 180px; color:#555; }
        .v { font-weight: 700; }
        button { padding: 10px 18px; border-radius: 12px; border: 1px solid #ddd; cursor: pointer; font-weight: 700; }
        .btn-start { background: #e7f7ee; }
        .btn-stop  { background: #fdecec; }
        .pill { padding: 8px 12px; border-radius: 999px; border: 1px solid #ddd; font-weight: 800; }
        .yes { background: #e7f7ee; }
        .no  { background: #fdecec; }
        .muted { color: #777; font-weight: 600; }
      </style>
    </head>
    <body>
      <h2>UNO WiFi Rev2 Sensor Dashboard</h2>

      <div class="card">
        <div class="row"><div class="k">Last update</div><div class="v" id="time">-</div></div>
        <div class="row"><div class="k">Temp (°C)</div><div class="v" id="temp">-</div></div>
        <div class="row"><div class="k">Humidity (%)</div><div class="v" id="hum">-</div></div>
        <div class="row"><div class="k">Soil raw</div><div class="v" id="soilraw">-</div></div>
        <div class="row"><div class="k">Soil moisture (%)</div><div class="v" id="soilpct">-</div></div>
        <div class="row"><div class="k">Light raw</div><div class="v" id="lightraw">-</div></div>
        <div class="row"><div class="k">Light level (%)</div><div class="v" id="lightpct">-</div></div>
        <div class="row"><div class="k">Light detected</div><div class="v" id="lightdet">-</div></div>

        <div class="row" style="margin-top: 16px;">
          <button class="btn-start" onclick="sendCmd('START')">START</button>
          <button class="btn-stop"  onclick="sendCmd('STOP')">STOP</button>

          <div class="muted" style="margin-left: 10px;">Irrigating:</div>
          <div class="pill" id="irrigatingPill">-</div>
        </div>
      </div>

      <script>
        function setIrrigatingPill(val){
          const pill = document.getElementById('irrigatingPill');
          const v = (val ?? '-').toString().toUpperCase();
          pill.textContent = v;
          pill.classList.remove('yes','no');
          if (v === 'YES') pill.classList.add('yes');
          else if (v === 'NO') pill.classList.add('no');
        }

        async function refresh(){
          const r = await fetch('/latest', {cache:'no-store'});
          const d = await r.json();
          document.getElementById('time').textContent = d.time ?? '-';
          document.getElementById('temp').textContent = d.temp_c ?? '-';
          document.getElementById('hum').textContent = d.humidity ?? '-';
          document.getElementById('soilraw').textContent = d.soil_raw ?? '-';
          document.getElementById('soilpct').textContent = d.soil_percent ?? '-';
          document.getElementById('lightraw').textContent = d.light_raw ?? '-';
          document.getElementById('lightpct').textContent = d.light_percent ?? '-';
          document.getElementById('lightdet').textContent = d.light_detected ?? '-';
          setIrrigatingPill(d.irrigating);
        }

        async function sendCmd(cmd){
          await fetch('/cmd', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({cmd})
          });
          // The hardware status updates when Arduino next uploads (~2s),
          // so we just refresh the page view.
          setTimeout(refresh, 250);
        }

        refresh();
        setInterval(refresh, 2000);
      </script>
    </body>
    </html>
    """
    return Response(html, mimetype="text/html")


def watchdog():
    global last_ingest_ts
    warned = False
    while True:
        time.sleep(CHECK_EVERY_SEC)
        if last_ingest_ts is None:
            continue
        gap = time.time() - last_ingest_ts
        if gap > TIMEOUT_SEC and not warned:
            print(f"\n⚠️  WARNING: No /ingest data received for {gap:.1f}s (Arduino/WiFi/power?)\n")
            sys.stdout.flush()
            warned = True
        if gap <= TIMEOUT_SEC:
            warned = False


if __name__ == "__main__":
    threading.Thread(target=watchdog, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)





