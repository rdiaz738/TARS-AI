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
import subprocess
import tkinter as tk
from PIL import ImageTk
import time  # Required for sleep in the extension thread

# Load configuration
CONFIG = load_config()
NEST_API_URL = "https://smartdevicemanagement.googleapis.com/v1"
app = Flask(__name__)  # Flask app for OAuth callback
auth_code = None  # Global variable to store the authorization code
qr_window = None  # Global variable to reference the QR code window
current_stream_extension_token = None  # To store the current stream extension token
stream_extension_thread = None  # To reference the extension thread
stream_extension_active = False  # Flag to control the extension thread

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
    global auth_code, qr_window
    auth_code = request.args.get("code")
    logging.info("Authorization successful! Closing QR code window.")
    
    # Close the Tkinter QR code window if it's open
    if qr_window:
        qr_window.quit()
        qr_window = None
    
    return "Authorization successful! You can close this tab."


def generate_qr_code(auth_url):
    """
    Generate and display a QR code in a Tkinter window.
    """
    global qr_window
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(auth_url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')

    # Initialize Tkinter window
    qr_window = tk.Tk()
    qr_window.title("Scan QR Code for Authentication")

    # Convert PIL image to Tkinter PhotoImage
    tk_img = ImageTk.PhotoImage(img)
    label = tk.Label(qr_window, image=tk_img)
    label.image = tk_img  # Keep a reference to avoid garbage collection
    label.pack()

    # Center the window
    qr_window.update_idletasks()
    width = qr_window.winfo_width()
    height = qr_window.winfo_height()
    x = (qr_window.winfo_screenwidth() // 2) - (width // 2)
    y = (qr_window.winfo_screenheight() // 2) - (height // 2)
    qr_window.geometry(f'{width}x{height}+{x}+{y}')

    # Run Tkinter in a separate thread to allow interaction
    Thread(target=qr_window.mainloop, daemon=True).start()


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
    Fetch a live stream URL from the Nest camera and store the stream extension token.
    """
    global current_stream_extension_token
    url = f"{NEST_API_URL}/{CONFIG['NEST']['device_id']}:executeCommand"
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {"command": "sdm.devices.commands.CameraLiveStream.GenerateRtspStream", "params": {}}
    logging.info(f"URL: {url}")

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        results = response.json().get("results", {})
        stream_url = results.get("streamUrls", {}).get("rtspUrl")
        current_stream_extension_token = results.get("streamExtensionToken")
        if stream_url:
            return stream_url
        else:
            raise Exception("Live stream URL not found.")
    else:
        log_error_and_raise("Failed to fetch live stream", response)


def play_live_stream(stream_url):
    """
    Play the live stream using FFplay or fallback to saving the stream.
    """
    try:
        logging.info(f"Attempting to play live stream with FFplay: {stream_url}")
        ffplay_cmd = [
            "ffplay", "-rtsp_transport", "tcp", stream_url
        ]
        subprocess.run(ffplay_cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"FFplay failed to open the RTSP stream. Error: {e}")
        logging.info("Attempting to save the stream as a file instead.")

        try:
            output_file = "nest_stream.mp4"
            ffmpeg_cmd = [
                "ffmpeg", "-rtsp_transport", "tcp", "-i", stream_url,
                "-c", "copy", output_file
            ]
            logging.info(f"Saving stream to {output_file} with command: {' '.join(ffmpeg_cmd)}")
            subprocess.run(ffmpeg_cmd, check=True)
            logging.info(f"Stream saved to {output_file}")
        except subprocess.CalledProcessError as ffmpeg_error:
            logging.error(f"FFmpeg also failed to save the RTSP stream. Error: {ffmpeg_error}")
        except FileNotFoundError:
            logging.error("FFmpeg is not installed. Please install it to use this functionality.")
    except FileNotFoundError:
        logging.error("FFplay is not installed. Please install FFmpeg to use this functionality.")

def extend_stream():
    """
    Periodically extend the RTSP stream before it expires.
    """
    global current_stream_extension_token, stream_extension_active
    while stream_extension_active:
        try:
            # Wait for 4 minutes before extending (to ensure it's before the 5-minute expiry)
            time.sleep(240)  # 4 minutes in seconds
            
            if not current_stream_extension_token:
                logging.warning("No stream extension token available to extend the stream.")
                continue
            
            access_token = get_access_token()
            device_id = CONFIG['NEST']['device_id']
            project_id = CONFIG['NEST']['project_id']
            url = f"{NEST_API_URL}/enterprises/{project_id}/devices/{device_id}:executeCommand"
            headers = {"Authorization": f"Bearer {access_token}"}
            payload = {
                "command": "sdm.devices.commands.CameraLiveStream.ExtendRtspStream",
                "params": {
                    "streamExtensionToken": current_stream_extension_token
                }
            }
            logging.info("Sending ExtendRtspStream command to extend the live stream.")
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                results = response.json().get("results", {})
                new_stream_extension_token = results.get("streamExtensionToken")
                if new_stream_extension_token:
                    current_stream_extension_token = new_stream_extension_token
                    logging.info("Successfully extended the live stream.")
                else:
                    logging.error("Failed to retrieve new stream extension token.")
            else:
                logging.error(f"Failed to extend the live stream. Status Code: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logging.error(f"Error while extending the stream: {e}")


def handle_nest_camera_live_stream():
    """
    Fetch and display the live stream from the Nest camera.
    """
    global stream_extension_thread, stream_extension_active
    try:
        access_token = CONFIG['NEST'].get("access_token")
        if not access_token:
            raise ValueError("Access token is missing.")

        # Get the live stream URL
        stream_url = get_camera_live_stream(access_token)
        logging.info(f"RTSP Stream URL: {stream_url}")
        play_live_stream(stream_url)

        # Start the stream extension thread
        if not stream_extension_active:
            stream_extension_active = True
            stream_extension_thread = Thread(target=extend_stream, daemon=True)
            stream_extension_thread.start()
    except requests.RequestException as req_err:
        logging.error(f"Failed to fetch the live stream due to a request error: {req_err}")
    except ValueError as val_err:
        logging.error(f"Error with live stream: {val_err}")
    except Exception as gen_err:
        logging.error(f"An unexpected error occurred: {gen_err}")

def stop_live_stream():
    """
    Stop the live stream and the extension thread.
    """
    global stream_extension_active, stream_extension_thread, current_stream_extension_token
    try:
        access_token = get_access_token()
        device_id = CONFIG['NEST']['device_id']
        project_id = CONFIG['NEST']['project_id']
        url = f"{NEST_API_URL}/enterprises/{project_id}/devices/{device_id}:executeCommand"
        headers = {"Authorization": f"Bearer {access_token}"}
        payload = {
            "command": "sdm.devices.commands.CameraLiveStream.StopRtspStream",
            "params": {
                "streamExtensionToken": current_stream_extension_token
            }
        }
        logging.info("Sending StopRtspStream command to stop the live stream.")
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            logging.info("Successfully stopped the live stream.")
        else:
            logging.error(f"Failed to stop the live stream. Status Code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        logging.error(f"Error while stopping the stream: {e}")
    finally:
        # Stop the extension thread
        stream_extension_active = False
        if stream_extension_thread and stream_extension_thread.is_alive():
            stream_extension_thread.join(timeout=1)
        stream_extension_thread = None
        current_stream_extension_token = None


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