import sys
import os
import subprocess
import time

# ==========================================
# 1. PRE-FLIGHT CHECK (Verify Keys)
# ==========================================
try:
    import keys
    print("[INIT] Credentials detected.")
    # We don't print the actual keys for security, just confirm they exist
    if not keys.ACCOUNT_SID or not keys.AUTH_TOKEN:
        print("[WARNING] keys.py is present but variables seem empty.")
except ImportError:
    print("[ERROR] 'keys.py' not found! Please create it in this folder.")
    print("The modules may fail to send SMS alerts.")
    time.sleep(2) # Give user time to read

# ==========================================
# 2. HELPER TO RUN MODULES
# ==========================================
def launch_module(script_path):
    """
    Runs a python script as a separate process.
    Passes the current environment so it can find keys.py
    """
    if not os.path.exists(script_path):
        print(f"[ERROR] File not found: {script_path}")
        return

    print(f"\n[LAUNCHING] Starting {script_path}...")
    print("------------------------------------------------")
    
    # Use sys.executable to ensure we use the same Python that runs main.py
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except KeyboardInterrupt:
        print("\n[STOPPED] Module stopped by user.")
    except Exception as e:
        print(f"\n[CRASH] The module crashed: {e}")
    
    print("------------------------------------------------")
    print("[INFO] Returning to Main Menu...")
    time.sleep(1)

# ==========================================
# 3. MAIN MENU
# ==========================================
def main():
    while True:
        # Clear screen (optional, works on Windows/Mac/Linux)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("========================================")
        print("   MULTIMODAL THREAT DETECTION SYSTEM   ")
        print("========================================")
        print("1. Start AUDIO Surveillance (KWS)")
        print("2. Start VISUAL Surveillance (Gesture)")
        print("3. Exit")
        print("========================================")
        
        choice = input("Select an option (1-3): ").strip()

        if choice == '1':
            # Path to your audio script
            path = os.path.join("audio_module", "kws.py")
            launch_module(path)
            
        elif choice == '2':
            # Path to your vision script
            path = os.path.join("vision_module", "gesture_recognizer.py")
            launch_module(path)
            
        elif choice == '3':
            print("Exiting System. Goodbye!")
            break
        else:
            print("Invalid selection. Try again.")
            time.sleep(1)

if __name__ == "__main__":
    main()