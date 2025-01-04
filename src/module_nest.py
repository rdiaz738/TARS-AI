import requests
from flask import Flask, request
from threading import Thread
from urllib.parse import urlencode
from PIL import Image
from io import BytesIO
import webbrowser
import logging
import qrcode  # For generating QR codes
import os  # For launching VLC/FFmpeg
from module_config import load_config

# Load configuration
CONFIG = load_config()
NEST_API_URL = "https://smartdevicemanagement.googleapis.com/v1"
app = Flask(__name__)  # Flask app for OAuth callback
auth_code = None  # Global variable to store the authorization code
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
print(f"Using Client ID: {CONFIG['NEST']['client_id']}")

# === Helper Functions ===
def log_error_and_raise(message, response=None):
    """
    Log an error message and raise an exception.
    """
    if response:
        logging.error(f"{message} - Status: {response.status_code}, Response: {response.text}")
    else:
        logging.error(message)
    raise Exception(message)

# === Authentication Flow ===
def get_auth_code_url():
    """
    Generate the Google OAuth 2.0 authorization URL.
    """
    params = {
        "client_id": CONFIG['NEST']['client_id'],
        "redirect_uri": CONFIG['NEST']["redirect_url"],
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/sdm.service"
    }
    return f"https://accounts.google.com/o/oauth2/auth?{urlencode(params)}"

@app.route("/callback")
def callback():
    """
    Handle the OAuth callback and extract the authorization code.
    """
    global auth_code
    auth_code = request.args.get("code")
    return "Authorization successful! You can close this tab."

def generate_qr_code(auth_url):
    """
    Generate a QR code for the authorization URL.
    """
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(auth_url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    img.show()

def exchange_code_for_tokens():
    """
    Exchange the authorization code for access and refresh tokens.
    """
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
    """
    Start the authentication flow with a local Flask server.
    """
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
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    CONFIG['NEST']['access_token'] = access_token
    if refresh_token:
        CONFIG['NEST']['refresh_token'] = refresh_token

    devices = list_nest_devices(access_token)
    logging.info("Access Token stored successfully.")
    return devices

# === Token Management ===
def get_access_token():
    """
    Retrieve the access token from the configuration.
    """
    access_token = CONFIG['NEST'].get('access_token')
    logging.info(f"ACCESS TOKEN: {access_token}")
    if not access_token:
        logging.error("No access token found in configuration.")
        raise Exception("Missing access token.")
    return access_token

# === Camera Snapshot and Live Stream ===
def get_camera_snapshot(access_token):
    """
    Fetch a snapshot from the Nest camera.
    """
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
    """
    Display the fetched snapshot using Pillow.
    """
    image = Image.open(BytesIO(image_bytes))
    image.show()

def fetch_and_display_snapshot():
    """
    Fetch and display a snapshot from the Nest camera.
    """
    try:
        access_token = get_access_token()
        image_bytes = get_camera_snapshot(access_token)
        display_snapshot(image_bytes)
    except Exception as e:
        logging.error(f"Failed to fetch and display snapshot: {e}")

def get_camera_live_stream(access_token):
    """
    Fetch a live stream URL from the Nest camera.
    """
    url = f"{NEST_API_URL}/{CONFIG['NEST']['device_id']}:executeCommand"
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {"command": "sdm.devices.commands.CameraLiveStream.GenerateRtspStream", "params": {}}
    logging.info(f"URL: {url}")

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        stream_url = response.json().get("results", {}).get("streamUrls", {}).get("rtspUrl")
        if stream_url:
            return stream_url
        else:
            raise Exception("Live stream URL not found.")
    else:
        log_error_and_raise("Failed to fetch live stream", response)

def play_live_stream(stream_url):
    """
    Play the live stream using VLC or FFmpeg.
    """
    try:
        logging.info(f"Playing live stream: {stream_url}")
        # Option 1: Use VLC
        os.system(f"vlc {stream_url}")

        # Option 2: Uncomment to save the stream using FFmpeg
        # os.system(f'ffmpeg -i "{stream_url}" -c copy output.mp4')
    except Exception as e:
        logging.error(f"Failed to play the live stream: {e}")

def handle_nest_camera_live_stream():
    """
    Fetch and display the live stream from the Nest camera.
    """
    try:
        access_token = get_access_token()
        stream_url = get_camera_live_stream(access_token)
        play_live_stream(stream_url)
    except Exception as e:
        logging.error(f"Failed to fetch and display live stream: {e}")

# === Device Management ===
def list_nest_devices(access_token):
    """
    List all devices associated with the Nest account and return them.
    """
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
            return []
    else:
        log_error_and_raise("Failed to fetch devices", response)

def validate_camera_device(devices, device_id, trait="sdm.devices.traits.CameraEventImage"):
    """
    Validate that the provided device ID corresponds to a camera and supports the required trait.
    """
    for device in devices:
        if device['name'] == device_id and trait in device.get('traits', {}):
            return True
    return False
