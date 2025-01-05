import requests
from flask import Flask, request
from threading import Thread, Event
from urllib.parse import urlencode
from PIL import Image, ImageTk
from io import BytesIO
import tkinter as tk
import logging
import qrcode  # For generating QR codes
import os  # For launching VLC/FFmpeg
from module_config import load_config
import subprocess
import time

# Load configuration
CONFIG = load_config()
NEST_API_URL = "https://smartdevicemanagement.googleapis.com/v1"
app = Flask(__name__)  # Flask app for OAuth callback
auth_code = None  # Global variable to store the authorization code
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logging.info(f"Using Client ID: {CONFIG['NEST']['client_id']}")

# Event to signal QR code window to close
qr_closed_event = Event()

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
        "scope": "https://www.googleapis.com/auth/sdm.service",
        "access_type": "offline",
        "prompt": "consent"
    }
    return f"https://nestservices.google.com/partnerconnections/{CONFIG['NEST']['project_id']}/auth?{urlencode(params)}"

@app.route("/callback")
def callback():
    """
    Handle the OAuth callback and extract the authorization code.
    """
    global auth_code
    auth_code = request.args.get("code")
    logging.info("Authorization code received.")
    qr_closed_event.set()  # Signal to close the QR code window
    return "Authorization successful! You can close this tab."

def generate_qr_code(auth_url):
    """
    Generate the QR code image.
    """
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(auth_url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    return img

def display_qr_code(img):
    """
    Display the QR code using Tkinter and close it upon authentication.
    """
    def close_window():
        root.destroy()

    root = tk.Tk()
    root.title("Scan QR Code to Authenticate")

    # Convert PIL image to Tkinter PhotoImage
    tk_img = ImageTk.PhotoImage(img)
    label = tk.Label(root, image=tk_img)
    label.pack(padx=20, pady=20)

    # Instruction Label
    instruction = tk.Label(root, text="Scan this QR code with your device to authenticate.")
    instruction.pack(pady=(0, 20))

    # Periodically check if the QR code should be closed
    def check_event():
        if qr_closed_event.is_set():
            close_window()
        else:
            root.after(1000, check_event)  # Check every second

    root.after(1000, check_event)
    root.mainloop()

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
        tokens = response.json()
        logging.info("Access and refresh tokens obtained successfully.")
        return tokens
    else:
        log_error_and_raise("Failed to exchange code for tokens", response)

def refresh_access_token():
    """
    Refresh the access token using the refresh token.
    """
    refresh_token = CONFIG['NEST'].get('refresh_token')
    if not refresh_token:
        log_error_and_raise("No refresh token available.")

    url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": CONFIG['NEST']['client_id'],
        "client_secret": CONFIG['NEST']['client_secret'],
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        tokens = response.json()
        CONFIG['NEST']['access_token'] = tokens.get('access_token')
        if tokens.get('refresh_token'):
            CONFIG['NEST']['refresh_token'] = tokens.get('refresh_token')
        logging.info("Access token refreshed successfully.")
        return tokens.get('access_token')
    else:
        log_error_and_raise("Failed to refresh access token", response)

def start_flask_app():
    """
    Start the Flask app to handle OAuth callback.
    """
    try:
        app.run(host="0.0.0.0", port=8080, debug=False)
    except Exception as e:
        logging.error(f"Flask server failed to start: {e}")

def start_auth_flow():
    """
    Start the authentication flow with a local Flask server and display the QR code.
    """
    global auth_code

    # Start Flask in a separate daemon thread
    flask_thread = Thread(target=start_flask_app, daemon=True)
    flask_thread.start()

    # Generate the authorization URL
    auth_url = get_auth_code_url()
    logging.info(f"Visit this URL to authenticate:\n{auth_url}")
    qr_image = generate_qr_code(auth_url)

    # Display the QR code in a separate daemon thread
    qr_thread = Thread(target=display_qr_code, args=(qr_image,), daemon=True)
    qr_thread.start()

    # Wait until the QR code window is closed (authentication is complete)
    qr_closed_event.wait()

    # Exchange the authorization code for tokens
    tokens = exchange_code_for_tokens()
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    CONFIG['NEST']['access_token'] = access_token
    if refresh_token:
        CONFIG['NEST']['refresh_token'] = refresh_token

    # Make the initial devices.list API call to complete authorization
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
        token = response.json().get("results", {}).get("token")
        if image_url and token:
            headers_image = {"Authorization": f"Basic {token}"}
            image_response = requests.get(image_url, headers=headers_image)
            if image_response.status_code == 200:
                return image_response.content
            else:
                log_error_and_raise("Failed to download snapshot image", image_response)
        else:
            raise Exception("Snapshot URL or token not found.")
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
    logging.info(f"Sending request to fetch live stream: {url}")

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        stream_urls = response.json().get("results", {}).get("streamUrls", {})
        rtsp_url = stream_urls.get("rtspUrl")
        if rtsp_url:
            return rtsp_url
        else:
            raise Exception("Live stream RTSP URL not found.")
    else:
        log_error_and_raise("Failed to fetch live stream", response)

def play_live_stream(stream_url):
    """
    Play the live stream using FFplay.
    """
    try:
        logging.info(f"Starting FFplay with RTSP URL: {stream_url}")
        ffplay_cmd = [
            "ffplay", "-rtsp_transport", "tcp", "-autoexit", "-fflags", "nobuffer", stream_url
        ]
        process = subprocess.Popen(ffplay_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return process
    except FileNotFoundError:
        logging.error("FFplay is not installed. Please install FFmpeg to use this functionality.")
        return None
    except Exception as e:
        logging.error(f"Failed to start FFplay: {e}")
        return None

def handle_nest_camera_live_stream():
    """
    Continuously fetch and display the live stream from the Nest camera.
    """
    try:
        access_token = get_access_token()
        devices = list_nest_devices(access_token)
        camera_device = CONFIG['NEST']['device_id']

        if not validate_camera_device(devices, camera_device):
            logging.error("Invalid camera device or missing required traits.")
            return

        while True:
            access_token = get_access_token()
            stream_url = get_camera_live_stream(access_token)
            logging.info(f"RTSP Stream URL: {stream_url}")

            # Start FFplay process
            ffplay_process = play_live_stream(stream_url)
            if not ffplay_process:
                logging.error("Failed to start FFplay. Retrying in 30 seconds...")
                time.sleep(30)
                continue

            # Wait for 4 minutes before extending the stream
            logging.info("Live stream started. Will extend in 4 minutes.")
            time.sleep(240)  # 4 minutes

            # Refresh the access token
            refresh_access_token()

            # Extend the RTSP stream by generating a new stream URL
            extended_stream_url = get_camera_live_stream(get_access_token())
            logging.info(f"Extended RTSP Stream URL: {extended_stream_url}")

            # Terminate the current FFplay process
            if ffplay_process.poll() is None:
                logging.info("Terminating current FFplay process to refresh stream.")
                ffplay_process.terminate()
                try:
                    ffplay_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logging.warning("FFplay did not terminate gracefully. Killing process.")
                    ffplay_process.kill()

            # Update the stream URL for the next iteration
            CONFIG['NEST']['stream_url'] = extended_stream_url

    except Exception as e:
        logging.error(f"An error occurred while handling live stream: {e}")

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
                device_name = device.get('name')
                device_type = device.get('type')
                display_name = device.get('traits', {}).get('Info', {}).get('customName', 'Unnamed Device')
                logging.info(f"Name: {device_name}, Type: {device_type}, Display Name: {display_name}")
            return devices
        else:
            logging.info("No devices found.")
            return []
    else:
        log_error_and_raise("Failed to fetch devices", response)

def validate_camera_device(devices, device_id, trait="sdm.devices.traits.CameraLiveStream"):
    """
    Validate that the provided device ID corresponds to a camera and supports the required trait.
    """
    for device in devices:
        if device['name'] == device_id and trait in device.get('traits', {}):
            return True
    return False

