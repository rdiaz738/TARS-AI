import requests
from flask import Flask, request
from threading import Thread
from urllib.parse import urlencode
from PIL import Image
from io import BytesIO
import logging
import qrcode
import subprocess
from module_config import load_config

# Load configuration
CONFIG = load_config()
NEST_API_URL = "https://smartdevicemanagement.googleapis.com/v1"
app = Flask(__name__)
auth_code = None
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
print(f"Using Client ID: {CONFIG['NEST']['client_id']}")

# === Helper Functions ===
def log_error_and_raise(message, response=None):
    if response:
        logging.error(f"{message} - Status: {response.status_code}, Response: {response.text}")
    else:
        logging.error(message)
    raise Exception(message)

def get_auth_code_url():
    params = {
        "client_id": CONFIG['NEST']['client_id'],
        "redirect_uri": CONFIG['NEST']["redirect_url"],
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/sdm.service"
    }
    return f"https://accounts.google.com/o/oauth2/auth?{urlencode(params)}"

@app.route("/callback")
def callback():
    global auth_code
    auth_code = request.args.get("code")
    return "Authorization successful! You can close this tab."

def generate_qr_code(auth_url):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(auth_url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    img.show()

def exchange_code_for_tokens():
    global auth_code
    if not auth_code:
        raise Exception("No authorization code found. Ensure the authentication flow is completed.")

    url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": CONFIG['NEST']['client_id'],
        "client_secret": CONFIG['NEST']['client_secret'],
        "code": auth_code,
        "grant_type": "authorization_code",
        "redirect_uri": CONFIG['NEST']["redirect_url"]
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        return response.json()
    else:
        log_error_and_raise("Failed to exchange code", response)

def start_auth_flow():
    global auth_code
    def run_flask():
        try:
            app.run(host="0.0.0.0", port=8080, debug=False)
        except Exception as e:
            logging.error(f"Flask server failed to start: {e}")

    Thread(target=run_flask).start()
    auth_url = get_auth_code_url()
    logging.info(f"Visit this URL to authenticate:\n{auth_url}")
    generate_qr_code(auth_url)

    while auth_code is None:
        pass

    tokens = exchange_code_for_tokens()
    CONFIG['NEST']['access_token'] = tokens.get("access_token")
    CONFIG['NEST']['refresh_token'] = tokens.get("refresh_token")

    logging.info("Access Token stored successfully.")
    list_nest_devices(CONFIG['NEST']['access_token'])

# === Token Management ===
def get_access_token():
    access_token = CONFIG['NEST'].get('access_token')
    if not access_token:
        logging.error("No access token found in configuration.")
        raise Exception("Missing access token.")
    return access_token

# === Camera Snapshot and Live Stream ===
def get_camera_snapshot(access_token):
    url = f"{NEST_API_URL}/{CONFIG['NEST']['device_id']}:executeCommand"
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {"command": "sdm.devices.commands.CameraEventImage.GenerateImage", "params": {}}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        image_url = response.json().get("results", {}).get("url")
        if image_url:
            return requests.get(image_url).content
        else:
            raise Exception("Snapshot URL not found.")
    else:
        log_error_and_raise("Failed to fetch snapshot", response)

def display_snapshot(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image.show()

def fetch_and_display_snapshot():
    try:
        access_token = get_access_token()
        image_bytes = get_camera_snapshot(access_token)
        display_snapshot(image_bytes)
    except Exception as e:
        logging.error(f"Failed to fetch and display snapshot: {e}")

def get_camera_live_stream(access_token):
    url = f"{NEST_API_URL}/{CONFIG['NEST']['device_id']}:executeCommand"
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {"command": "sdm.devices.commands.CameraLiveStream.GenerateRtspStream", "params": {}}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("results", {}).get("streamUrls", {}).get("rtspUrl")
    else:
        log_error_and_raise("Failed to fetch live stream", response)

def play_live_stream(stream_url):
    try:
        subprocess.run(["ffplay", "-rtsp_transport", "tcp", stream_url], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"FFplay failed. Error: {e}")
    except FileNotFoundError:
        logging.error("FFplay not found. Please install FFmpeg.")

# === Device Management ===
def list_nest_devices(access_token):
    url = f"{NEST_API_URL}/enterprises/{CONFIG['NEST']['project_id']}/devices"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        devices = response.json().get("devices", [])
        if devices:
            logging.info("Available Devices:")
            for device in devices:
                logging.info(f"Name: {device.get('name')}, Type: {device.get('type')}")
            return devices
        else:
            logging.info("No devices found.")
    else:
        log_error_and_raise("Failed to fetch devices", response)
