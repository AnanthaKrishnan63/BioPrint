import time
import csv
import json
import platform
import psutil
import shutil
import socket
from screeninfo import get_monitors
import threading
from pynput import keyboard, mouse
from scapy.all import sniff, wrpcap
import os
import subprocess
import sys
import winreg
from datetime import datetime
import wmi
import atexit
import numpy as np

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

folder_name = f"data_{timestamp}"
os.makedirs(folder_name, exist_ok=True)

# Path to ffmpeg
def get_ffmpeg_path():
    return os.path.join(os.path.dirname(__file__), "ffmpeg", "bin", "ffmpeg.exe")

CONSENT_FILE = os.path.join(folder_name, f"consent_granted_{timestamp}.txt")

# Function for consent
def show_consent_form():
    if os.path.exists(CONSENT_FILE):
        with open(CONSENT_FILE, "r") as f:
            for line in f:
                if line.startswith("User:"):
                    return line.split(":")[1].strip()  
    
    name = input("Please enter your name: ").strip()

    print(f"Hello {name}, please read the consent form below.")
    print("Consent Form:")
    print("By using this software, you consent to the collection of the following data:")
    print("1. Keystrokes")
    print("2. Mouse movements and clicks")
    print("3. Network traffic data")
    print("4. Hardware configuration")
    print("5. Valorant Configuration File")

    consent = input("Do you consent to this data collection? (yes/no): ").strip().lower()

    if consent == 'yes':
        with open(CONSENT_FILE, 'w') as f:
            f.write(f"User: {name}\nConsented: Yes\nTimestamp: {datetime.now()}")
        print("Thank you for consenting.")
        return name  # Return the name
    else:
        print("You must consent to proceed.")
        return None

# Get user name 
user_name = show_consent_form()
if not user_name:
    exit()

LOG_FILE = os.path.join(folder_name, f"keyboard_{user_name}_{timestamp}.csv")
MOUSE_LOG_FILE = os.path.join(folder_name, f"mouse_{user_name}_{timestamp}.csv")
PCAP_FILE = os.path.join(folder_name, f"captured_packets_{user_name}_{timestamp}.pcap")
HARDWARE_INFO_FILE = os.path.join(folder_name, f"hardware_info_{user_name}_{timestamp}.json")
DESTINATION_CONFIG_FOLDER = os.path.join(folder_name, f"valorant_config_{user_name}_{timestamp}")
SCREEN_RECORD_FILE = os.path.join(folder_name, f"screen_record_{user_name}_{timestamp}.mp4") 

# Initialize variables
click_count = 0
double_click_count = 0
scroll_count = 0
last_click_time = 0
click_start_time = 0
click_positions = []
key_press_times = {}
keyboard_last_time = None
start_time = time.time()
exit_keys = {keyboard.Key.ctrl_l, keyboard.Key.esc} 
active_keys_list = []  

# Get screen information
monitor = get_monitors()[0]
aspect_ratio = f"{monitor.width}:{monitor.height}"

csv_file = open(MOUSE_LOG_FILE, 'w', newline='') 
csv_writer = csv.writer(csv_file)


#Check for npcap installation
def is_npcap_installed():
    
    npcap_path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "Npcap", "Packet.dll")
    if os.path.exists(npcap_path):
        return True
    
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Npcap")
        winreg.CloseKey(key)
        return True
    except FileNotFoundError:
        return False
    

def get_installer_path():

    if getattr(sys, 'frozen', False):
        #Running as an executable
        base_path = sys._MEIPASS 
    else:
        # Running as a script
        base_path = os.path.dirname(os.path.abspath(__file__))

    installer_path = os.path.join(base_path, "npcap_installation_exe", "npcap-1.80.exe")
    
    return installer_path


#Install npcap
def install_npcap():
    
    installer_path = get_installer_path()

    if not os.path.exists(installer_path):
        print(f"Npcap installer not found at {installer_path}! Ensure the installer is placed correctly.")
        sys.exit(1)

    print("Npcap not found. Installing...")

    try:
        subprocess.run([
            "powershell",
            "Start-Process", installer_path,
            "-ArgumentList", "/silent",
            "-Wait",
            "-Verb", "RunAs"
        ], shell=True, check=True)

        print("Npcap installation completed.")
    except subprocess.CalledProcessError as e:
        print(f"Npcap installation failed: {e}")
        sys.exit(1)

def restart_script_without_admin():
    print("Restarting script without admin privileges...")
    
    script = sys.executable
    args = sys.argv
    if " " in script:
        script = f'"{script}"'
    
    subprocess.run(f"powershell Start-Process {script} -ArgumentList {' '.join(args)}", shell=True)
    sys.exit(0)


# Installing screen capturer recorder
def is_screen_recorder_installed():
    try:
        ffmpeg_path = get_ffmpeg_path()
        result = subprocess.run(
            [ffmpeg_path, "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True
        )
        # print(result.stderr)
        return "screen-capture-recorder" in result.stderr
    except Exception as e:
        print(f"Error checking devices: {e}")
        return False
    
def run_installer():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(__file__)

    installer_path = os.path.join(base_path, 'screen-capturer-recorder', 'screen-capturer-installer.exe')

    if not os.path.exists(installer_path):
        print("Installer not found.")
        return

    print("Launching screen capturer installer...")

    # Launch as admin
    subprocess.run([
        'powershell',
        'Start-Process', installer_path, '-Verb', 'runAs'
    ])

    # Check again if installed
    if is_screen_recorder_installed():
        print("Screen Capturer Recorder installed successfully.")
    else:
        print("Installation might have failed.")

 


def get_mac_address():
    try:
        mac_addresses = []
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == psutil.AF_LINK:  
                    mac_addresses.append(addr.address)
        return mac_addresses if mac_addresses else "Unknown"
    except:
        return "Unknown"
    
def get_hardware_info():
    
    info = {
        "OS": platform.system(),
        "OS_Version": platform.version(),
        "OS_Release": platform.release(),
        "Architecture": platform.architecture()[0],
        "Processor": platform.processor(),
        "RAM": f"{psutil.virtual_memory().total // (1024 ** 3)} GB",
        "Machine": platform.machine(),
        "Hostname": socket.gethostname(),
        "IP_Address": socket.gethostbyname(socket.gethostname()),
        "MAC_Address":  get_mac_address(),
        "Monitors": []
    }
    
    for monitor in get_monitors():
        info["Monitors"].append({
            "Width": monitor.width,
            "Height": monitor.height,
            "Aspect_Ratio": f"{monitor.width}:{monitor.height}"
        })
    
    return info

def save_hardware_info():
    hardware_info = get_hardware_info()
    with open(HARDWARE_INFO_FILE, 'w') as json_file:
        json.dump(hardware_info, json_file, indent=4)

# Start hardware collection before other listeners
save_hardware_info()


# Device categories to extract
TARGET_DEVICES = {
    "Keyboard": "Keyboard",
    "Mouse": "Mouse",
    "Monitor": "Monitor",
    "Speaker": "Audio",
    "Bluetooth": "Bluetooth"
}

def get_target_devices():
    c = wmi.WMI()
    devices = c.Win32_PnPEntity()

    target_device_info = {category: [] for category in TARGET_DEVICES.keys()}

    for device in devices:
        if not device.PNPDeviceID:  # Ignore devices without a valid ID
            continue

        device_name = device.Name.lower() if device.Name else "unknown"  # Handle NoneType
        device_data = {
            "Name": device.Name if device.Name else "Unknown",
            "HID": device.PNPDeviceID,
            "Status": device.Status if device.Status else "Unknown",
            "DeviceID": device.DeviceID if device.DeviceID else "Unknown"
        }

        # Classify device based on name
        for category, keyword in TARGET_DEVICES.items():
            if keyword.lower() in device_name:
                target_device_info[category].append(device_data)
                break  # Avoid duplicate classifications

    return target_device_info

def save_target_devices():
    all_devices = get_target_devices()
    with open(HARDWARE_INFO_FILE, "r+", encoding="utf-8") as json_file:
        data = json.load(json_file)
        data["Target_I/O_Devices"] = all_devices  # Update with filtered devices
        json_file.seek(0)
        json.dump(data, json_file, indent=4)

# Call function to collect and save device data
save_target_devices()

USERNAME = os.getenv("USERNAME")
BASE_CONFIG_PATH = f"C:\\Users\\{USERNAME}\\AppData\\Local\\VALORANT\\Saved\\Config"


def get_valorant_config():
    try:
        if not os.path.exists(BASE_CONFIG_PATH):
            print("Valorant config path does not exist.")
            return

        unique_folders = [f for f in os.listdir(BASE_CONFIG_PATH) if os.path.isdir(os.path.join(BASE_CONFIG_PATH, f))]

        if not unique_folders:
            print("No unique folder found in Valorant config path.")
            return

        
        UNIQUE_CONFIG_PATH = os.path.join(BASE_CONFIG_PATH)

        if not os.path.exists(UNIQUE_CONFIG_PATH):
            print(f"Config folder '{UNIQUE_CONFIG_PATH}' not found.")
            return

        
        if os.path.exists(DESTINATION_CONFIG_FOLDER):
            shutil.rmtree(DESTINATION_CONFIG_FOLDER)

        shutil.copytree(UNIQUE_CONFIG_PATH, DESTINATION_CONFIG_FOLDER)

        print(f"Valorant config folder copied successfully to '{DESTINATION_CONFIG_FOLDER}'.")

    except Exception as e:
                print(f"Error retrieving Valorant config folder: {e}")
get_valorant_config()





def initialize_csv():
    # Initialize CSV for keylogger
    with open(LOG_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([ 
            "Start Time", "Release Time", "Key", "Elapsed Start Time", 
            "Elapsed Release Time", "Duration", "Polling Rate", "Consecutive Keys"
        ])
    
    # Initialize CSV for mouse tracker
    csv_writer.writerow([
        'Timestamp', 'X', 'Y', 'Event', 'Click Count', 'Double Click Count', 
        'Scroll Count', 'Speed', 'Movement Distance', 'Movement Time', 'Acceleration', 'Elapsed Time'
    ])

# Keyboard logging function
def log_to_csv_keyboard(start_press_time, release_time, key, elapsed_start, elapsed_release, duration=None, polling_rate=None, consecutive_keys=None):
    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([ 
            start_press_time,
            release_time,
            key.lower(),
            f"{elapsed_start:.3f}" if elapsed_start else "",
            f"{elapsed_release:.3f}" if elapsed_release else "",
            f"{duration:.3f}" if duration else 0.0,
            f"{polling_rate:.3f}" if polling_rate else 0.0,
            consecutive_keys if consecutive_keys else ""
        ])

# Keyboard listener callbacks
def on_key_press(key):
    global keyboard_last_time
    current_time = time.time()
    polling_rate = (current_time - keyboard_last_time) if keyboard_last_time else 0.0
    keyboard_last_time = current_time
    elapsed_start_time = (time.time() - start_time)

    try:
        key_char = key.char.lower()
    except AttributeError:
        key_char = str(key).lower()

    if key_char not in active_keys_list:
        active_keys_list.append(key_char)

    key_press_times[key] = {
        "start_press_time": time.time(),
        "elapsed_start_time": elapsed_start_time,
        "polling_rate": polling_rate
    }

    if exit_keys.issubset(set(active_keys_list)):
        return False 
def on_key_release(key):
    global current_keys
    release_time = time.time()
    elapsed_release_time = (release_time - start_time)

    try:
        key_char = key.char.lower()
    except AttributeError:
        key_char = str(key).lower()

    if key in key_press_times:
        press_details = key_press_times.pop(key)
        duration = (release_time - press_details["start_press_time"])
        polling_rate = press_details["polling_rate"]
        start_press_time = press_details["start_press_time"]
        elapsed_start_time = press_details["elapsed_start_time"]

        log_to_csv_keyboard(
            start_press_time,
            release_time,
            key_char,
            elapsed_start_time,
            elapsed_release_time,
            duration=duration,
            polling_rate=polling_rate,
            consecutive_keys=", ".join(active_keys_list)
        )

    if key_char in active_keys_list:
        active_keys_list.remove(key_char)

    if exit_keys.issubset(set(active_keys_list)):
        return False  

# Mouse tracking functions
def calculate_speed(x1, y1, x2, y2, time_diff):
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance / time_diff if time_diff > 0 else 0

def calculate_acceleration(prev_speed, curr_speed, time_diff):
    return (curr_speed - prev_speed) / time_diff if time_diff > 0 else 0

last_position = None
last_move_time = None
last_speed = 0

# Mouse listener callbacks with exit condition
def on_move(x, y):
    global last_position, last_move_time, last_speed
    current_time = time.time()

    if exit_keys.issubset(set(active_keys_list)):
        return False  
    if last_position is not None:
        time_diff = current_time - last_move_time
        speed = calculate_speed(last_position[0], last_position[1], x, y, time_diff)
        acceleration = calculate_acceleration(last_speed, speed, time_diff)
        distance = ((x - last_position[0])**2 + (y - last_position[1])**2)**0.5
        elapsed_time = current_time - start_time

        csv_writer.writerow([
            current_time, x, y, 'Move', click_count, double_click_count, scroll_count, 
            speed, distance, time_diff, acceleration, elapsed_time
        ])

        last_speed = speed

    last_position = (x, y)
    last_move_time = current_time

def on_click(x, y, button, pressed):
    global click_count, double_click_count, last_click_time, click_start_time, click_positions
    current_time = time.time()

 
    if exit_keys.issubset(set(active_keys_list)):
        return False  

    event_type = f'Click {button}' if pressed else f'Release {button}'
    click_count += 1 if pressed else 0
    click_start_time = current_time if pressed else click_start_time

    csv_writer.writerow([
        current_time, x, y, event_type, click_count, double_click_count, scroll_count, '', '', '', '', (current_time - start_time)
    ])

def on_scroll(x, y, dx, dy):
    global scroll_count
    scroll_count += 1
    
    if exit_keys.issubset(set(active_keys_list)):
        return False

    event_type = 'Scroll Up' if dy > 0 else 'Scroll Down' if dy < 0 else 'Scroll Horizontal'
    csv_writer.writerow([
        time.time(), x, y, event_type, click_count, double_click_count, scroll_count, '', '', '', '', (time.time() - start_time)
    ])

# Packet capture listener
def packet_capture_callback(packet):
   
    if exit_keys.issubset(set(active_keys_list)):
        return False 

    wrpcap(PCAP_FILE, packet, append=True)

# Start listeners
def start_listeners():
    with keyboard.Listener(on_press=on_key_press, on_release=on_key_release) as keyboard_listener:
        keyboard_listener.join()

def start_mouse_listener():
    with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as mouse_listener:
        mouse_listener.join()

def start_packet_capture_listener():
    sniff(prn=packet_capture_callback, store=0, count=0)  


# Screen recording
def record_screen(output_file=SCREEN_RECORD_FILE):
    global recording_process

    # Get full screen resolution dynamically
    resolution = aspect_ratio
    # print("first check")
    ffmpeg_path = get_ffmpeg_path() 

    # print("second check")
    command = [
        ffmpeg_path,
        "-f", "dshow",  
        "-i", "video=screen-capture-recorder",
        "-framerate", "25",  
        "-video_size", resolution,  
        # "-i", "desktop",  
        # "-vf", "scale=854:720", 
        "-c:v", "libx264",  
        "-preset", "ultrafast",  
        "-pix_fmt", "yuv420p",  
        output_file  
    ]

    recording_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    



def main():

    if not is_npcap_installed():
        install_npcap()
        restart_script_without_admin()

    if not is_screen_recorder_installed():
        run_installer()    

    #Initialise logging csv
    initialize_csv()  

    # Start listeners
    keyboard_thread = threading.Thread(target=start_listeners, daemon=True)
    mouse_thread = threading.Thread(target=start_mouse_listener, daemon=True)
    packet_capture_thread = threading.Thread(target=start_packet_capture_listener, daemon=True)
    screen_thread = threading.Thread(target=record_screen, daemon=True)

    # Start threads
    keyboard_thread.start()
    mouse_thread.start()
    packet_capture_thread.start()
    screen_thread.start()

    # Keep main thread alive to avoid premature exit
    keyboard_thread.join()
    mouse_thread.join()
    packet_capture_thread.join()

if __name__ == "__main__":
    main()