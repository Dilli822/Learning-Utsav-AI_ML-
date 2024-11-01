import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()

# Set the speech rate to normal speed
engine.setProperty('rate', 150)

# Function to speak text
def speak_text(text):
    engine.say(text)  # Add the text to the queue
    engine.runAndWait()  # Block while processing all currently queued commands

# Function to read predefined text from a .txt file
def read_predefined_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print("Error: The predefined text file was not found.")
        return None

# Function for real-time user input for TTS
def real_time_tts():
    print("Choose an option:")
    print("1: Read predefined text from a file")
    print("2: Type your own text")
    
    option = input("Enter 1 or 2: ")

    if option == '1':
        # Read predefined text from a file
        predefined_text = read_predefined_text('story.txt')
        if predefined_text:
            print("Speaking predefined text...")
            speak_text(predefined_text)
    
    elif option == '2':
        print("Type your text and press Enter. Type 'exit' to quit.")
        user_input = ""
        
        while True:
            # Take user input
            input_text = input()
            
            # Check for exit command
            if input_text.lower() == 'exit':
                print("Goodbye!")
                break
            
            # Append input text to user input
            user_input += input_text + " "  # Add a space for clarity
            
            # Speak the text immediately after user presses Enter
            if input_text:  # Only speak if input is not empty
                # Remove punctuation from the user input before speaking
                clean_text = user_input.strip().rstrip('.?!')
                print("Speaking...")
                speak_text(clean_text)  # Speak the cleaned text
                user_input = ""  # Clear the user input after speaking
    else:
        print("Invalid option. Please restart the program and choose either 1 or 2.")

# Run the real-time TTS function
real_time_tts()
