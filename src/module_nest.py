import requests
from urllib.parse import urlencode
from PIL import Image
from io import BytesIO
import time
from module_config import load_config

# Load configuration
CONFIG = load_config()
NEST_API_URL = "https://smartdevicemanagement.googleapis.com/v1"

def get_auth_code_url():
    """
    Generate the Google OAuth 2.0 authorization URL.
    """
    params = {
        "client_id": CONFIG['NEST']['client_id'],
        "redirect_uri": "http://localhost",
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/sdm.service"
    }
    return f"https://accounts.google.com/o/oauth2/auth?{urlencode(params)}"

def exchange_code_for_tokens(auth_code):
    """
    Exchange the authorization code for access and refresh tokens.
    """
    url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": CONFIG['NEST']['client_id'],
        "client_secret": CONFIG['NEST']['client_secret'],
        "code": auth_code,
        "grant_type": "authorization_code",
        "redirect_uri": "http://localhost"
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to exchange code: {response.text}")

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

def get_camera_snapshot(access_token):
    """
    Fetches a snapshot from the Nest camera.
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
    Displays the fetched snapshot using Pillow.
    """
    image = Image.open(BytesIO(image_bytes))
    image.show()

def fetch_and_display_snapshot():
    """
    Continuously fetches and displays snapshots at regular intervals.
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

def initiate_auth_flow():
    """
    Initiates the OAuth authentication flow for Nest API.
    """
    print("Visit the following URL to authenticate:")
    print(get_auth_code_url())
    auth_code = input("Enter the authorization code from the URL: ").strip()
    try:
        tokens = exchange_code_for_tokens(auth_code)
        print("Access Token:", tokens['access_token'])
        print("Refresh Token:", tokens['refresh_token'])
    except Exception as e:
        print(f"Error during authentication: {e}")
