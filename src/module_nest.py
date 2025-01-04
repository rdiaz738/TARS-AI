import requests
from flask import Flask, request, redirect
from threading import Thread
from urllib.parse import urlencode
from PIL import Image
from io import BytesIO
import time
from module_config import load_config

# Load configuration
CONFIG = load_config()
NEST_API_URL = "https://smartdevicemanagement.googleapis.com/v1"
app = Flask(__name__)  # Flask app for OAuth callback
auth_code = None  # Global variable to store the authorization code
print(f"Using Client ID: {CONFIG['NEST']['client_id']}")

# === Authentication Flow ===
def get_auth_code_url():
    """
    Generate the Google OAuth 2.0 authorization URL.
    """
    params = {
        "client_id": CONFIG['NEST']['client_id'],
        "redirect_uri": "http://localhost:8080/callback",
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
        "redirect_uri": "http://localhost:8080/callback"
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to exchange code: {response.text}")

def start_auth_flow():
    """
    Start the authentication flow with a local Flask server.
    """
    global auth_code

    # Start Flask app in a separate thread
    def run_flask():
        app.run(port=8080, debug=False)

    Thread(target=run_flask).start()

    # Open the authentication URL in the browser
    print("Visit this URL to authenticate:")
    print(get_auth_code_url())

    # Wait for the auth code to be set
    while auth_code is None:
        pass

    # Exchange the code for tokens
    tokens = exchange_code_for_tokens()
    list_nest_devices(tokens.get("access_token"))
    print("Access Token:", tokens.get("access_token"))
    print("Refresh Token:", tokens.get("refresh_token"))

# === Token Management ===
def refresh_access_token():
    """
    Refresh the access token using the refresh token.
    """
    url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": CONFIG['NEST']['client_id'],
        "client_secret": CONFIG['NEST']['client_secret'],
        "refresh_token": CONFIG['NEST']['refresh_token'],
        "grant_type": "refresh_token"
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception(f"Failed to refresh access token: {response.text}")

# === Nest Camera Operations ===
def get_camera_snapshot(access_token):
    """
    Fetch a snapshot from the Nest camera.
    """
    url = f"{NEST_API_URL}/enterprises/{CONFIG['NEST']['project_id']}/devices/{CONFIG['NEST']['device_id']}:executeCommand"
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
        raise Exception(f"Failed to fetch snapshot: {response.text}")

def display_snapshot(image_bytes):
    """
    Display the fetched snapshot using Pillow.
    """
    image = Image.open(BytesIO(image_bytes))
    image.show()

def fetch_and_display_snapshot():
    """
    Continuously fetch and display snapshots at regular intervals.
    """
    try:
        while True:
            access_token = refresh_access_token()
            image_bytes = get_camera_snapshot(access_token)
            display_snapshot(image_bytes)
            time.sleep(30)
    except Exception as e:
        print(f"Error: {e}")

def list_nest_devices(access_token):
    """
    List all devices associated with the Nest account.
    """
    url = f"{NEST_API_URL}/enterprises/{CONFIG['NEST']['project_id']}/devices"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        devices = response.json().get("devices", [])
        if devices:
            print("Available Devices:")
            for device in devices:
                print(f"Name: {device.get('name')}")
                print(f"Type: {device.get('type')}")
                print(f"Traits: {device.get('traits')}")
        else:
            print("No devices found.")
    else:
        print(f"Error fetching devices: {response.text}")
