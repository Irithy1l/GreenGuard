from datetime import datetime
from functools import wraps
from pathlib import Path
import io
import os
import sqlite3
import time

from flask import Flask, jsonify, redirect, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import torch
    from torchvision import transforms
except Exception:
    torch = None
    transforms = None

app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
DB_PATH = Path(__file__).with_name("temp.db")

command_state = {"cmd": "STOP"}

# Auto-watering policy (tuned for container plants; see notes in response).
AUTO_WATERING_ENABLED = True
SOIL_DRY_TRIGGER_PERCENT = 35.0
WATERING_DURATION_SECONDS = 20
MIN_SECONDS_BETWEEN_WATERING = 3600
REALTIME_SENSOR_MAX_AGE_SECONDS = 20

auto_watering_state = {
    "watering_until_ts": None,
    "last_watering_start_ts": None,
}

_nn_model = None
_nn_metadata = None
_nn_device = None


def _load_nn_model():
    global _nn_model, _nn_metadata, _nn_device
    if _nn_model is not None:
        return _nn_model, _nn_metadata, _nn_device

    if torch is None or transforms is None:
        raise RuntimeError("PyTorch/torchvision not available on server")

    ckpt_path = Path("models") / "soil_water_classifier.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"Model checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    classes = checkpoint.get("classes", ["DO_NOT_WATER", "WATER"])
    input_size = checkpoint.get("input_size", [128, 128])
    mean = checkpoint.get("normalize_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("normalize_std", [0.229, 0.224, 0.225])

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, len(classes)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _nn_model = model
    _nn_metadata = {
        "classes": classes,
        "input_size": input_size,
        "mean": mean,
        "std": std,
    }
    _nn_device = torch.device("cpu")
    return _nn_model, _nn_metadata, _nn_device


def run_plant_model(image_bytes):
    if Image is None:
        raise RuntimeError("Pillow is not available on server")

    model, meta, device = _load_nn_model()
    classes = meta["classes"]
    input_h, input_w = int(meta["input_size"][0]), int(meta["input_size"][1])

    preprocess = transforms.Compose(
        [
            transforms.Resize((input_h, input_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=meta["mean"], std=meta["std"]),
        ]
    )

    with Image.open(io.BytesIO(image_bytes)).convert("RGB") as img:
        x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)

    pred_idx = int(idx.item())
    pred_label = classes[pred_idx] if 0 <= pred_idx < len(classes) else "DO_NOT_WATER"
    return {
        "model": "soil_water_classifier.pt",
        "recommendation": pred_label,
        "confidence": float(conf.item()),
        "note": "Prediction from trained PyTorch CNN model.",
    }


def analyze_uploaded_plant_image(image_bytes):
    if not image_bytes:
        return {"ok": False, "error": "empty file"}, 400

    if len(image_bytes) > app.config["MAX_CONTENT_LENGTH"]:
        return {"ok": False, "error": "file too large"}, 413

    if Image is not None:
        try:
            with Image.open(io.BytesIO(image_bytes)) as im:
                im.verify()
        except Exception:
            return {"ok": False, "error": "invalid image"}, 400

    result = run_plant_model(image_bytes)
    payload = {
        "ok": True,
        "recommendation": result.get("recommendation", "DO_NOT_WATER"),
        "confidence": float(result.get("confidence", 0.0)),
        "model": result.get("model", "placeholder-v0"),
        "note": result.get("note", ""),
    }
    return payload, 200


def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema():
    with db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                device_name TEXT NOT NULL,
                register_date TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_data (
                data_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ts TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                value REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watered_dates_by_user (
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                PRIMARY KEY (user_id, date),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """
        )

        # One-time migration from legacy watered_dates(date) to per-user table (user_id=1).
        legacy_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='watered_dates'"
        ).fetchone()
        if legacy_table:
            conn.execute(
                """
                INSERT OR IGNORE INTO watered_dates_by_user(user_id, date)
                SELECT 1, date FROM watered_dates
                """
            )

        # Migration for older users table without register_date.
        user_cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        if "register_date" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN register_date TEXT")
            conn.execute(
                "UPDATE users SET register_date = date('now', 'localtime') WHERE register_date IS NULL OR register_date = ''"
            )

        conn.commit()


def parse_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_sensor_ts(ts_value):
    try:
        return datetime.strptime(ts_value, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _stop_watering_if_expired(now_ts=None):
    if now_ts is None:
        now_ts = time.time()
    until = auto_watering_state["watering_until_ts"]
    if until is not None and now_ts >= until:
        command_state["cmd"] = "STOP"
        auto_watering_state["watering_until_ts"] = None
        return True
    return False


def _evaluate_auto_watering(soil_percent):
    if not AUTO_WATERING_ENABLED:
        return

    now_ts = time.time()
    _stop_watering_if_expired(now_ts)

    if soil_percent is None:
        return
    if command_state["cmd"] == "START":
        return
    if soil_percent > SOIL_DRY_TRIGGER_PERCENT:
        return

    last_start = auto_watering_state["last_watering_start_ts"]
    if last_start is not None and (now_ts - last_start) < MIN_SECONDS_BETWEEN_WATERING:
        return

    command_state["cmd"] = "START"
    auto_watering_state["last_watering_start_ts"] = now_ts
    auto_watering_state["watering_until_ts"] = now_ts + WATERING_DURATION_SECONDS


def current_user_id():
    uid = session.get("user_id")
    return int(uid) if uid else None


def login_required_api(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if current_user_id() is None:
            return jsonify({"ok": False, "error": "authentication required"}), 401
        return fn(*args, **kwargs)

    return wrapped


def login_required_page(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if current_user_id() is None:
            return redirect(url_for("login_page"))
        return fn(*args, **kwargs)

    return wrapped


def resolve_ingest_user_id():
    username = (request.args.get("username") or "").strip()
    if username:
        with db_connect() as conn:
            row = conn.execute(
                "SELECT user_id FROM users WHERE username = ?", (username,)
            ).fetchone()
            if row:
                return int(row["user_id"])

    user_id = request.args.get("user_id", type=int)
    if user_id and user_id > 0:
        return user_id

    return 1


def ingest_rows(user_id, ts, payload):
    mappings = [
        ("temp_c", "temperature"),
        ("humidity", "humidity"),
        ("soil_percent", "soil_moisture"),
        ("soil_raw", "soil_raw"),
        ("light_percent", "light_level"),
        ("light_raw", "light_raw"),
    ]

    rows = []
    for source_key, sensor_type in mappings:
        parsed = parse_float(payload.get(source_key))
        if parsed is not None:
            rows.append((user_id, ts, sensor_type, parsed))

    irrigating = str(payload.get("irrigating", "")).strip().upper()
    if irrigating in ("YES", "NO"):
        rows.append((user_id, ts, "irrigating", 1.0 if irrigating == "YES" else 0.0))

    if not rows:
        return False, False

    watered_today = irrigating == "YES"
    with db_connect() as conn:
        conn.executemany(
            """
            INSERT INTO sensor_data(user_id, ts, sensor_type, value)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        if watered_today:
            conn.execute(
                "INSERT OR IGNORE INTO watered_dates_by_user(user_id, date) VALUES (?, ?)",
                (user_id, ts[:10]),
            )
        conn.commit()

    return True, watered_today


@app.post("/api/register")
def api_register():
    data = request.get_json(force=True, silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()
    device_name = str(data.get("device_name", "")).strip() or f"{username}-device"
    register_date = datetime.now().strftime("%Y-%m-%d")

    if not username or not password:
        return jsonify({"ok": False, "error": "username and password are required"}), 400

    hashed_password = generate_password_hash(password)
    try:
        with db_connect() as conn:
            cur = conn.execute(
                "INSERT INTO users(username, password, device_name, register_date) VALUES (?, ?, ?, ?)",
                (username, hashed_password, device_name, register_date),
            )
            conn.commit()
            user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "username already exists"}), 409

    session["user_id"] = user_id
    session["username"] = username
    session["register_date"] = register_date
    return jsonify({"ok": True, "user_id": user_id, "username": username})


@app.post("/api/login")
def api_login():
    data = request.get_json(force=True, silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()

    if not username or not password:
        return jsonify({"ok": False, "error": "username and password are required"}), 400

    with db_connect() as conn:
        row = conn.execute(
            "SELECT user_id, username, password, register_date FROM users WHERE username = ?", (username,)
        ).fetchone()

        if not row:
            return jsonify({"ok": False, "error": "invalid username or password"}), 401

        stored = row["password"]
        valid = False

        # Backward compatibility with any existing plaintext rows.
        if stored and stored.startswith(("pbkdf2:", "scrypt:")):
            valid = check_password_hash(stored, password)
        else:
            valid = stored == password
            if valid:
                conn.execute(
                    "UPDATE users SET password = ? WHERE user_id = ?",
                    (generate_password_hash(password), row["user_id"]),
                )
                conn.commit()

        if not valid:
            return jsonify({"ok": False, "error": "invalid username or password"}), 401

    session["user_id"] = int(row["user_id"])
    session["username"] = row["username"]
    session["register_date"] = row["register_date"]
    return jsonify({"ok": True, "user_id": int(row["user_id"]), "username": row["username"]})


@app.post("/api/logout")
def api_logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/me")
def api_me():
    uid = current_user_id()
    if uid is None:
        return jsonify({"ok": False, "authenticated": False}), 401
    register_date = session.get("register_date")
    if not register_date:
        with db_connect() as conn:
            row = conn.execute(
                "SELECT register_date FROM users WHERE user_id = ?",
                (uid,),
            ).fetchone()
        register_date = row["register_date"] if row else None
        session["register_date"] = register_date
    return jsonify(
        {
            "ok": True,
            "authenticated": True,
            "user_id": uid,
            "username": session.get("username"),
            "register_date": register_date,
        }
    )


@app.post("/ingest")
def ingest():
    payload = request.get_json(force=True, silent=True) or {}
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_id = resolve_ingest_user_id()
    soil_percent = parse_float(payload.get("soil_percent"))
    _evaluate_auto_watering(soil_percent)

    inserted, watered = ingest_rows(user_id=user_id, ts=ts, payload=payload)
    return jsonify(
        {
            "ok": True,
            "inserted": inserted,
            "watered_today": watered,
            "ts": ts,
            "user_id": user_id,
            "auto_watering": {
                "enabled": AUTO_WATERING_ENABLED,
                "soil_trigger_percent": SOIL_DRY_TRIGGER_PERCENT,
                "watering_duration_seconds": WATERING_DURATION_SECONDS,
                "current_cmd": command_state["cmd"],
            },
        }
    )


@app.get("/cmd")
def get_cmd():
    _stop_watering_if_expired()
    return jsonify(command_state)


@app.post("/cmd")
def set_cmd():
    _stop_watering_if_expired()
    data = request.get_json(force=True, silent=True) or {}
    cmd = str(data.get("cmd", "")).strip().upper()
    if cmd not in ("START", "STOP"):
        return jsonify({"ok": False, "error": "cmd must be START or STOP"}), 400
    command_state["cmd"] = cmd
    if cmd == "STOP":
        auto_watering_state["watering_until_ts"] = None
    else:
        now_ts = time.time()
        auto_watering_state["last_watering_start_ts"] = now_ts
        auto_watering_state["watering_until_ts"] = now_ts + WATERING_DURATION_SECONDS
    return jsonify({"ok": True, "cmd": cmd})


@app.get("/latest")
@login_required_api
def latest():
    _stop_watering_if_expired()
    user_id = current_user_id()
    sensor_types = ("temperature", "humidity", "soil_moisture", "light_level", "irrigating")
    placeholders = ",".join("?" for _ in sensor_types)

    with db_connect() as conn:
        rows = conn.execute(
            f"""
            SELECT s.sensor_type, s.value, s.ts
            FROM sensor_data s
            JOIN (
                SELECT sensor_type, MAX(ts) AS max_ts
                FROM sensor_data
                WHERE user_id = ? AND sensor_type IN ({placeholders})
                GROUP BY sensor_type
            ) latest_row
              ON s.sensor_type = latest_row.sensor_type AND s.ts = latest_row.max_ts
            WHERE s.user_id = ?
            """,
            (user_id, *sensor_types, user_id),
        ).fetchall()

    now = datetime.now()
    result = {
        "user_id": user_id,
        "time": None,
        "data_fresh": False,
        "data_age_seconds": None,
        "realtime_window_seconds": REALTIME_SENSOR_MAX_AGE_SECONDS,
    }
    freshest_age = None
    fresh_irrigating = None

    for row in rows:
        if result["time"] is None or row["ts"] > result["time"]:
            result["time"] = row["ts"]

        parsed_ts = parse_sensor_ts(row["ts"])
        if not parsed_ts:
            continue
        age_seconds = (now - parsed_ts).total_seconds()
        if freshest_age is None or age_seconds < freshest_age:
            freshest_age = max(0, int(age_seconds))

        is_fresh = age_seconds <= REALTIME_SENSOR_MAX_AGE_SECONDS
        if not is_fresh:
            continue

        if row["sensor_type"] == "irrigating":
            fresh_irrigating = "YES" if row["value"] >= 0.5 else "NO"
        else:
            result[row["sensor_type"]] = row["value"]

    result["data_age_seconds"] = freshest_age
    result["data_fresh"] = freshest_age is not None and freshest_age <= REALTIME_SENSOR_MAX_AGE_SECONDS

    # For live watering state, do not trust stale DB rows.
    if fresh_irrigating is not None:
        result["irrigating"] = fresh_irrigating
        result["irrigating_source"] = "sensor_realtime"
    else:
        result["irrigating"] = "YES" if command_state["cmd"] == "START" else "NO"
        result["irrigating_source"] = "server_cmd"

    return jsonify(result)


@app.get("/api/watered-dates")
@login_required_api
def api_watered_dates():
    user_id = current_user_id()
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT date FROM watered_dates_by_user WHERE user_id = ? ORDER BY date ASC",
            (user_id,),
        ).fetchall()
    return jsonify([row["date"] for row in rows])


@app.get("/api/sensor-data")
@login_required_api
def api_sensor_data():
    user_id = current_user_id()
    hours = request.args.get("hours", default=72, type=int)
    if hours < 1:
        hours = 1

    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT ts, sensor_type, value
            FROM sensor_data
            WHERE user_id = ?
              AND ts >= datetime('now', 'localtime', ?)
              AND sensor_type IN ('temperature', 'humidity', 'soil_moisture', 'light_level', 'irrigating')
            ORDER BY ts ASC
            """,
            (user_id, f"-{hours} hours"),
        ).fetchall()

    return jsonify([dict(row) for row in rows])


@app.post("/api/ai-water-check")
@login_required_api
def api_ai_water_check():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "image file is required"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    payload, status = analyze_uploaded_plant_image(image_bytes)
    return jsonify(payload), status


@app.get("/ai-check")
@login_required_page
def ai_check_page():
    return send_from_directory("static", "ai_check.html")


@app.get("/login")
def login_page():
    if current_user_id() is not None:
        return redirect(url_for("index"))
    return send_from_directory("static", "auth.html")


@app.get("/")
@login_required_page
def index():
    return send_from_directory("static", "index.html")


@app.get("/sensor")
@login_required_page
def sensor_page():
    return send_from_directory("static", "sensor.html")


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


if __name__ == "__main__":
    ensure_schema()
    app.run(host="0.0.0.0", port=5000, debug=True)
