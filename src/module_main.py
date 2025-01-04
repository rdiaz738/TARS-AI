"""
module_main.py

Core logic module for the TARS-AI application.

Integrates modules and manages key functionalities, including:
- Text-to-Speech (TTS) playback and configuration
- Interaction with Large Language Model (LLM) backends
- Prompt building and AI response processing
- Wake word handling and user interaction workflows
- Emotion detection and system threading
- Nest Camera Integration
"""

# === Standard Libraries ===
import os
import threading
import json
import requests
import re
from datetime import datetime
import concurrent.futures

# === Custom Modules ===
from module_config import load_config
from module_btcontroller import start_controls
from module_engine import check_for_module
from module_tts import generate_tts_audio
from module_vision import get_image_caption_from_base64
from module_stt import STTManager
from module_nest import start_auth_flow, refresh_access_token, get_camera_snapshot, display_snapshot, get_camera_live_stream, display_live_stream, list_nest_devices

# === Constants and Globals ===
character_manager = None
memory_manager = None
stt_manager = None

CONFIG = load_config()

# Global Variables (if needed)
stop_event = threading.Event()
executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)

# === Nest Camera Authentication ===
def start_nest_auth_flow():
    """
    Start the OAuth flow for Nest Camera authentication.
    """
    print("Starting Nest authentication flow...")
    start_auth_flow()

# === Nest Camera Snapshot Handling ===
def handle_nest_camera_snapshot():
    """
    Fetch and display a snapshot from the Nest camera.
    """
    try:
        print("[INFO] Fetching access token...")
        access_token = refresh_access_token()

        print("[INFO] Validating device traits...")
        devices = list_nest_devices(access_token)
        if not validate_camera_device(devices, CONFIG['NEST']['device_id']):
            print("[ERROR] Device ID does not correspond to a camera.")
            return

        print("[INFO] Fetching snapshot...")
        image_bytes = get_camera_snapshot(access_token)

        print("[INFO] Displaying snapshot...")
        display_snapshot(image_bytes)
    except Exception as e:
        print(f"[ERROR] Failed to display snapshot: {e}")

# === Threads ===
def start_bt_controller_thread():
    """
    Wrapper to start the BT Controller functionality in a thread.
    """
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LOAD: Starting BT Controller thread...")
        while not stop_event.is_set():
            start_controls()
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Error in BT Controller thread: {e}")
# === Get Nest Devices ===
def fetch_nest_devices():
    """
    Fetch and list available Nest devices.
    """
    if CONFIG["NEST"].getboolean("use_nest"):
        print("Fetching available Nest devices...")
        access_token = refresh_access_token()
        if access_token:
            list_nest_devices(access_token)
    else:
        print("Nest Camera functionality is disabled in the configuration.")
# === Display nest live ===
def handle_nest_camera_live_stream():
    """
    Fetch and display the live stream from the Nest camera.
    """
    try:
        print("[INFO] Fetching access token...")
        access_token = refresh_access_token()

        print("[INFO] Validating device traits...")
        devices = list_nest_devices(access_token)
        if not validate_camera_device(devices, CONFIG['NEST']['device_id']):
            print("[ERROR] Device ID does not correspond to a camera.")
            return

        print("[INFO] Fetching live stream URL...")
        stream_url = get_camera_live_stream(access_token)

        print("[INFO] Displaying live stream...")
        display_live_stream(stream_url)
    except Exception as e:
        print(f"[ERROR] Failed to display live stream: {e}")
# === Core Functions ===
def extract_text(json_response, picture):
    """
    Extracts text from the JSON response. Handles OpenAI's chat.completion and other structures.

    Parameters:
    - json_response (dict): The JSON response from the LLM backend.
    - picture (bool): Whether the response contains a picture or not.

    Returns:
    - str: The extracted text content from the response.
    """
    global character_manager
    
    try:
        # Determine the correct field for text extraction based on response structure
        if 'choices' in json_response:
            if CONFIG['LLM']['llm_backend'] == "openai":
                # For OpenAI's chat.completion API
                text_content = json_response['choices'][0]['message']['content']
                return text_content
            elif CONFIG['LLM']['llm_backend'] == "ooba" or CONFIG['LLM']['llm_backend'] == "tabby":
                # For other backends like Ooba or Tabby
                text_content = json_response['choices'][0]['text']
        else:
            raise KeyError("Invalid response format: 'choices' key not found.")

        # Clean up the text
        cleaned_text = re.sub(r"\s{2,}", " ", text_content.strip())  # Collapse multiple spaces
        cleaned_text = re.sub(r"<\|.*?\|>", "", cleaned_text, flags=re.DOTALL)  # Remove <|...|> tags
        
        if not picture:
            # Additional cleanup for non-picture responses
            cleaned_text = re.sub(rf"{re.escape(character_manager.char_name)}:\s*", "", cleaned_text)  # Remove character name prefix
            cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text).strip()  # Remove empty lines

        return cleaned_text

    except (KeyError, IndexError, TypeError) as error:
        return f"Text content could not be found. Error: {str(error)}"

def set_emotion(text_to_read):
    """
    Function to set the emotion of the character based on the text generated by the AI.

    Parameters:
    - text_to_read (str): The text generated by the AI.
    """
    from transformers import pipeline
    global memory_manager
    
    sizecheck = memory_manager.token_count(text_to_read)
    if 'length' in sizecheck:
        value_to_convert = sizecheck['length']
    
    if isinstance(value_to_convert, (int, float)):
        if value_to_convert <= 511:
            classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
            model_outputs = classifier(text_to_read)
            emotion = max(model_outputs[0], key=lambda x: x['score'])['label']
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Emotion {emotion}")

def llm_process(userinput, botresponse):
    """
    Process the user input and bot response for various tasks.

    Parameters:
    - userinput (str): The user input text.
    - botresponse (str): The bot response text.

    Returns:
    - str: The processed bot response
    """
    global memory_manager

    threading.Thread(target=memory_manager.write_longterm_memory, args=(userinput, botresponse)).start()
    if CONFIG['EMOTION']['enabled'] == True: #set emotion
        threading.Thread(target=set_emotion, args=(botresponse,)).start()
    return botresponse

def build_prompt(user_prompt):
    """
    Build the prompt structure for the Large Language Model (LLM) backend.

    Parameters:
    - user_prompt (str): The user's input prompt.

    Returns:
    - str: The formatted prompt for the LLM backend.
    """
    global character_manager, memory_manager
    
    now = datetime.now() # Current date and time
    date = now.strftime("%m/%d/%Y")
    time = now.strftime("%H:%M:%S")

    # Handle toggling voice-only mode
    if "voice only mode on" in user_prompt:
        character_manager.voice_only = True
    elif "voice only mode off" in user_prompt:
        character_manager.voice_only = False

    module_engine = check_for_module(user_prompt)

    if module_engine == "Mute":
        #somehow needs to go back to listen for wake word
        return

    if module_engine != "No_Tool":
        #if "*User is leaving the chat politely*" in module_engine:
            #stop_idle() #StopAFK mssages

        if "Sends a picture***" in module_engine:
            sdpicture = module_engine.split('***', 1)[-1]
            #module_engine = f"*Sends a picture*. You will inform user that this is the image as requested, do not describe the image."
            
            pattern = r'data:image\/[a-zA-Z]+;base64,([^"]+)'
            match = re.search(pattern, sdpicture)
            if match:
                base64_data = match.group(1)
                module_engine = f"*Sends a picture of: {get_image_caption_from_base64(base64_data)}*"
            else:
                module_engine = f"*Cannot send a picture something went wrong, inform user*"
 
    # Build basic prompt structure
    dtg = f"Current Date: {date}\nCurrent Time: {time}\n"
    past = memory_manager.get_longterm_memory(user_prompt) # Get past memories
    # Correct the order and logic of replacements clean up memories and past json crap
    past = past.replace("\\\\", "\\")  # Reduce double backslashes to single
    past = past.replace("\\n", "\n")   # Replace escaped newline characters with actual newlines
    past = past.replace("\\'", "'")    # Replace escaped single quotes with actual single quotes
    past = past.replace("\'", "'")    # Replace escaped single quotes with actual single quotes

    history = ""
    userInput = user_prompt  # Simulating user input to avoid hanging

    if module_engine != "No_Tool":
        module_engine = module_engine + "\n"
    else:
        module_engine = ""

    promptsize = (
        f"System: {CONFIG['LLM']['systemprompt']}\n\n"
        f"### Instruction: {CONFIG['LLM']['instructionprompt']}\n"
        f"{dtg}\n"
        f"User is: {CONFIG['CHAR']['user_details']}\n\n"
        f"{character_manager.character_card}\n"
        f"Past Memories which may be helpful to answer {character_manager.char_name}: {past}\n\n"
        f"{history}\n"
        #f"{module_engine}"
        f"Respond to {CONFIG['CHAR']['user_name']}'s message of: {userInput}\n"
        f"{module_engine}"
        f"### Response: {character_manager.char_name}: "
    )
    # Calc how much space is avail for chat history
    remaining = memory_manager.token_count(promptsize).get('length', 0)
    memallocation = int(CONFIG['LLM']['contextsize'] - remaining)
    history = memory_manager.get_shortterm_memories_tokenlimit(memallocation)

    prompt = (
        f"System: {CONFIG['LLM']['systemprompt']}\n\n"
        f"### Instruction: {CONFIG['LLM']['instructionprompt']}\n"
        f"{dtg}\n"
        f"User is: {CONFIG['CHAR']['user_details']}\n\n"
        f"{character_manager.character_card}\n"
        f"Past Memories which may be helpfull to answer {character_manager.char_name}: {past}\n\n"
        f"{history}\n"
        f"Respond to {CONFIG['CHAR']['user_name']}'s message of: {userInput}\n"
        f"{module_engine}"
        f"### Response: {character_manager.char_name}: "
    )
    prompt = prompt.replace("{user}", CONFIG['CHAR']['user_name']) 
    prompt = prompt.replace("{char}", CONFIG['CHAR']['user_name'])
    prompt = prompt.replace("\\\\", "\\") 
    prompt = prompt.replace("\\n", "\n") 
    prompt = prompt.replace("\\'", "'")    
    prompt = prompt.replace("\'", "'")    
    prompt = prompt.replace('\\"', '"')
    prompt = prompt.replace('\"', '"')
    prompt = prompt.replace('<END>', '') 

    return prompt

def get_completion(prompt, istext):
    """
    Get the completion from the LLM backend.

    Parameters:
    - prompt (str): The prompt to send to the LLM backend.
    - istext (str): Whether the prompt is text or not.

    Returns:
    - str: The generated completion
    """
    # Check if the prompt is text or not
    if istext == "True":
        prompt = build_prompt(prompt)

    # Set the header for the request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CONFIG['LLM']['api_key']}"
    }

    # Handle OpenAI backend
    if CONFIG['LLM']['llm_backend'] == "openai":
        url = f"{CONFIG['LLM']['base_url']}/v1/chat/completions"
        data = {
            "model": CONFIG['LLM']['openai_model'],  # GPT-4 or GPT-3.5-turbo
            "messages": [
                {"role": "system", "content": CONFIG['LLM']['systemprompt']},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": CONFIG['LLM']['max_tokens'],
            "temperature": CONFIG['LLM']['temperature'],
            "top_p": CONFIG['LLM']['top_p']
        }
    # Handle Ooba backend
    elif CONFIG['LLM']['llm_backend'] == "ooba":
        url = f"{CONFIG['LLM']['base_url']}/v1/completions"
        data = {
            "prompt": prompt,
            "max_tokens": CONFIG['LLM']['max_tokens'],
            "temperature": CONFIG['LLM']['temperature'],
            "top_p": CONFIG['LLM']['top_p'],
            "seed": CONFIG['LLM']['seed']
        }
    # Handle Tabby backend
    elif CONFIG['LLM']['llm_backend'] == "tabby":
        url = f"{CONFIG['LLM']['base_url']}/v1/completions"
        data = {
            "prompt": prompt,
            "max_tokens": CONFIG['LLM']['max_tokens'],
            "temperature": CONFIG['LLM']['temperature'],
            "top_p": CONFIG['LLM']['top_p']
        }
    else:
        raise ValueError(f"Unsupported LLM backend: {CONFIG['LLM']['llm_backend']}")

    # Send the request and get the response
    response = requests.post(url, headers=headers, data=json.dumps(data))
    try:
        response.raise_for_status()  # Handle HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: LLM request failed: {e}")
        return None  # Return None for failed requests

    # Check if the response is successful
    if istext == "False":
        text_to_read = extract_text(response.json(), True)
    else:
        text_to_read = extract_text(response.json(), False)
    text_to_read = text_to_read.replace('<END>', '') # Without this if may continue on forever (max token)

    return(text_to_read)

def process_completion(text):
    """
    Process the user input and generate a response using the Large Language Model (LLM) backend.

    Parameters:
    - text (str): The user input text.

    Returns:
    - str: The AI-generated response.
    """
    # Use the executor directly without 'with' statement
    future = executor.submit(get_completion, text, "True")
    botres = future.result()
    reply = llm_process(text, botres)
    return reply

# === Callback Functions ===
def wake_word_callback(wake_response):
    """
    Play initial response when wake word is detected.

    Parameters:
    - wake_response (str): The response to the wake word.
    """
    generate_tts_audio(wake_response, CONFIG['TTS']['ttsoption'], CONFIG['TTS']['azure_api_key'], CONFIG['TTS']['azure_region'], CONFIG['TTS']['ttsurl'], CONFIG['TTS']['toggle_charvoice'], CONFIG['TTS']['tts_voice'])

def utterance_callback(message):
    """
    Process the recognized message from STTManager and stream audio response to speakers.

    Parameters:
    - message (str): The recognized message from the Speech-to-Text (STT) module.
    """
    try:
        # Parse the user message
        message_dict = json.loads(message)
        if not message_dict.get('text'):  # Handles cases where text is "" or missing
            #print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TARS: Going Idle...")
            return
        #Print the response
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] USER: {message_dict['text']}")

        # Check for shutdown command
        if "shutdown pc" in message_dict['text'].lower():
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SHUTDOWN: Shutting down the PC...")
            os.system('shutdown /s /t 0')
            return  # Exit function after issuing shutdown command
        
        # Process the message using process_completion
        reply = process_completion(message_dict['text'])  # Process the message

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TARS: {reply}")
        # Stream TTS audio to speakers
        #print("Fetching TTS audio...")
        generate_tts_audio(reply, CONFIG['TTS']['ttsoption'], CONFIG['TTS']['azure_api_key'], CONFIG['TTS']['azure_region'], CONFIG['TTS']['ttsurl'], CONFIG['TTS']['toggle_charvoice'], CONFIG['TTS']['tts_voice'])

    except json.JSONDecodeError:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Invalid JSON format. Could not process user message.")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {e}")

def post_utterance_callback():
    """
    Restart listening for another utterance after handling the current one.
    """
    global stt_manager
    stt_manager._transcribe_utterance()

# === Initialization ===
def initialize_managers(mem_manager, char_manager, stt_mgr):
    """
    Pass in the shared instances for MemoryManager, CharacterManager, and STTManager.
    
    Parameters:
    - mem_manager: The MemoryManager instance from app.py.
    - char_manager: The CharacterManager instance from app.py.
    - stt_mgr: The STTManager instance from app.py.
    """
    global memory_manager, character_manager, stt_manager
    memory_manager = mem_manager
    character_manager = char_manager
    stt_manager = stt_mgr
start_nest_auth_flow()