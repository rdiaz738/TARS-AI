import requests
from flask import Flask, request
from threading import Thread, Event
from urllib.parse import urlencode
from PIL import Image, ImageTk
import tkinter as tk
import logging
import qrcode
import os
from module_config import load_config  # Ensure this module provides the necessary configuration
import subprocess
import time

# Load configuration
CONFIG = load_config()
NEST_API_URL = "https://smartdevicemanagement.googleapis.com/v1"
app = Flask(__name__)  # Flask app for OAuth callback
auth_code = None  # Global variable to store the authorization code

# Setup logging
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

def validate_camera_device(devices, device_id, required_traits=None):
    """
    Validate that the provided device ID corresponds to a camera and supports the required traits.
    """
    if required_traits is None:
        required_traits = ["sdm.devices.traits.CameraLiveStream"]
    for device in devices:
        if device['name'] == device_id:
            device_traits = device.get('traits', {})
            if all(trait in device_traits for trait in required_traits):
                logging.info(f"Device {device_id} is a valid camera with required traits.")
                return True
            else:
                missing = [trait for trait in required_traits if trait not in device_traits]
                logging.error(f"Device {device_id} is missing traits: {missing}")
                return False
    logging.error(f"Device {device_id} not found in the device list.")
    return False

# === Camera Snapshot and Live Stream ===
def get_camera_snapshot(access_token):
    """
    Fetch a snapshot from the Nest camera.
    """
    url = f"{NEST_API_URL}/{CONFIG['NEST']['device_id']}:executeCommand"
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {
        "command": "sdm.devices.commands.CameraEventImage.GenerateImage",
        "params": {}
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        image_url = response.json().get("results", {}).get("url")
        if image_url:
            # The token is required for authorization
            token = response.json().get("results", {}).get("token")
            headers_image = {"Authorization": f"Basic {token}"}
            image_response = requests.get(image_url, headers=headers_image)
            if image_response.status_code == 200:
                return image_response.content
            else:
                log_error_and_raise("Failed to download snapshot image", image_response)
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

def generate_rtsp_stream(access_token):
    """
    Generate an RTSP stream URL using the CameraLiveStream trait.
    """
    url = f"{NEST_API_URL}/{CONFIG['NEST']['device_id']}:executeCommand"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    payload = {
        "command": "sdm.devices.commands.CameraLiveStream.GenerateRtspStream",
        "params": {}
    }
    logging.info(f"Sending GenerateRtspStream command to {url}")
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        stream_url = response.json().get("results", {}).get("rtspUrl")
        stream_extension_token = response.json().get("results", {}).get("streamExtensionToken")
        stream_token = response.json().get("results", {}).get("streamToken")
        if stream_url and stream_extension_token and stream_token:
            logging.info("RTSP stream generated successfully.")
            return {
                "stream_url": stream_url,
                "stream_extension_token": stream_extension_token,
                "stream_token": stream_token
            }
        else:
            raise Exception("Incomplete stream information received.")
    else:
        log_error_and_raise("Failed to generate RTSP stream", response)

def extend_rtsp_stream(access_token, stream_extension_token):
    """
    Extend the RTSP stream before it expires.
    """
    url = f"{NEST_API_URL}/{CONFIG['NEST']['device_id']}:executeCommand"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    payload = {
        "command": "sdm.devices.commands.CameraLiveStream.ExtendRtspStream",
        "params": {
            "streamExtensionToken": stream_extension_token
        }
    }
    logging.info(f"Sending ExtendRtspStream command to {url}")
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        new_stream_extension_token = response.json().get("results", {}).get("streamExtensionToken")
        new_stream_token = response.json().get("results", {}).get("streamToken")
        if new_stream_extension_token and new_stream_token:
            logging.info("RTSP stream extended successfully.")
            return {
                "stream_extension_token": new_stream_extension_token,
                "stream_token": new_stream_token
            }
        else:
            raise Exception("Incomplete stream extension information received.")
    else:
        log_error_and_raise("Failed to extend RTSP stream", response)

def stop_rtsp_stream(access_token, stream_extension_token):
    """
    Stop the RTSP stream when it's no longer needed.
    """
    url = f"{NEST_API_URL}/{CONFIG['NEST']['device_id']}:executeCommand"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    payload = {
        "command": "sdm.devices.commands.CameraLiveStream.StopRtspStream",
        "params": {
            "streamExtensionToken": stream_extension_token
        }
    }
    logging.info(f"Sending StopRtspStream command to {url}")
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        logging.info("RTSP stream stopped successfully.")
    else:
        log_error_and_raise("Failed to stop RTSP stream", response)

# === Live Stream Handling ===
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
    except Exception as e:
        logging.error(f"Failed to start FFplay: {e}")
        return None

def play_live_stream_continuously():
    """
    Continuously play the live stream by refreshing the RTSP URL before it expires.
    """
    process = None
    stream_info = None
    try:
        while True:
            access_token = get_access_token()
            stream_info = generate_rtsp_stream(access_token)
            stream_url = stream_info["stream_url"]
            stream_extension_token = stream_info["stream_extension_token"]
            stream_token = stream_info["stream_token"]

            # Start FFplay process
            process = play_live_stream(stream_url)
            if not process:
                logging.error("FFplay process could not be started.")
                break

            # Wait for 4 minutes before extending the stream (assuming 5 minutes validity)
            logging.info("Stream started. Will attempt to extend in 4 minutes.")
            time.sleep(240)  # 4 minutes

            # Refresh the access token before extending the stream
            refresh_access_token()

            # Extend the RTSP stream
            extended_info = extend_rtsp_stream(access_token, stream_extension_token)
            new_stream_extension_token = extended_info["stream_extension_token"]
            new_stream_token = extended_info["stream_token"]

            # Terminate the current FFplay process
            if process:
                logging.info("Terminating current FFplay process to refresh stream.")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logging.warning("FFplay did not terminate gracefully. Killing process.")
                    process.kill()

            # Update stream_extension_token for the next iteration
            stream_extension_token = new_stream_extension_token

    except KeyboardInterrupt:
        logging.info("Stream playback interrupted by user.")
        if process:
            process.terminate()
    except Exception as e:
        logging.error(f"An error occurred during live stream playback: {e}")
        if process:
            process.terminate()
    finally:
        if stream_info and 'stream_extension_token' in stream_info:
            stop_rtsp_stream(access_token, stream_info['stream_extension_token'])

def handle_nest_camera_live_stream():
    """
    Continuously fetch and display the live stream from the Nest camera.
    """
    try:
        access_token = get_access_token()
        device_id = CONFIG['NEST']['device_id']

        # Validate device
        if validate_camera_device(devices, device_id):
            logging.info(f"Camera device {device_id} is valid.")
            # Start continuous playback in a separate thread
            stream_thread = Thread(target=play_live_stream_continuously, daemon=True)
            stream_thread.start()

            # Keep the main thread alive while the stream is playing
            while stream_thread.is_alive():
                stream_thread.join(timeout=1)
        else:
            logging.error("Invalid camera device or missing required traits.")
    except Exception as e:
        logging.error(f"Failed to handle live stream: {e}")

# === Main Execution ===
if __name__ == "__main__":
    try:
        devices = start_auth_flow()
        if devices:
            # For simplicity, select the first camera device
            camera_devices = [device for device in devices if device.get('type') == "sdm.devices.types.CAMERA"]
            if not camera_devices:
                logging.error("No camera devices found in your account.")
                exit(1)

            # Select the first camera device
            camera_device = camera_devices[0]['name']
            CONFIG['NEST']['device_id'] = camera_device

            # Validate device
            if validate_camera_device(devices, camera_device):
                logging.info(f"Camera device {camera_device} is valid.")
                handle_nest_camera_live_stream()
            else:
                logging.error("Invalid camera device or missing required traits.")
        else:
            logging.error("No devices available to stream.")
    except Exception as e:
        logging.error(f"An error occurred in the main execution flow: {e}")
