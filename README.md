# AI Fitness Coach

An interactive, data-driven fitness coach powered by Streamlit, MediaPipe, and Google's Gemini Pro. This application provides real-time biomechanical feedback on user workouts by analyzing joint angles and comparing them to ideal form criteria.



## Key Features

-   **Real-time Pose Estimation:** Utilizes MediaPipe to detect and track 33 body landmarks from a live camera feed or a pre-recorded video.
-   **Data-Driven Analysis:** Compares user's joint angles against a configurable `config.json` file defining ideal form for various exercises.
-   **Live Visual Feedback:** Overlays joint angles directly onto the video feed, color-coded (green, yellow, red) to indicate deviation from ideal form.
-   **Quantitative Data Table:** Displays a live-updating table with ideal vs. actual angles and their deviation for each metric.
-   **AI-Powered Coaching:** Sends the processed video and a quantitative performance summary to the Gemini Pro model for hyper-specific, actionable feedback.
-   **Voice Commands & Audio Feedback:** Uses a local Whisper model for "start workout" voice activation and gTTS for audio playback of the AI coach's feedback.
-   **Robust Testing Mode:** Allows for rapid testing and development using local video files, complete with a tool to correct video orientation.

---

## Setup and Installation

Follow these steps to get the AI Fitness Coach running on your local machine.

### Prerequisites

-   Python 3.9+
-   `git` for cloning the repository.
-   An environment variable for your Google API key.

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    -   **macOS / Linux:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    -   **Windows:**
        ```bash
        python -m venv .venv
        .\.venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    Install all required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first time you run the app, the Whisper speech recognition model (`~150MB`) will be downloaded and cached automatically.*

4.  **Set Up Google API Key:**
    This application requires a Google API key with the Gemini API enabled. You must set this key as an environment variable named `GOOGLE_API_KEY`.
    -   **macOS / Linux:**
        ```bash
        export GOOGLE_API_KEY="your_api_key_here"
        ```
        *(To make this permanent, add the line to your `~/.zshrc` or `~/.bash_profile`)*

    -   **Windows (Command Prompt):**
        ```bash
        set GOOGLE_API_KEY="your_api_key_here"
        ```

5.  **Configure Workouts:**
    Ensure the `config.json` file is present in the root directory. You can customize this file to add new exercises or tweak the ideal angles and thresholds for existing ones.

---

## Usage

1.  **Run the Streamlit App:**
    Make sure your virtual environment is activated and the `GOOGLE_API_KEY` is set. Then, run the following command:
    ```bash
    streamlit run app.py
    ```

2.  **Using the App:**
    -   **Live Mode:** The default mode. The app will use your webcam.
        -   Select a workout from the dropdown menu.
        -   Enable or disable voice commands with the toggle.
        -   Start the set by saying "start workout" or by clicking the manual start button.
        -   Perform the exercise, watching the real-time angle overlays and data table for feedback.
        -   After the set, the app will analyze your performance and provide both text and audio feedback from the AI coach.
    -   **Testing Mode:**
        -   Toggle "Testing Mode" on.
        -   Select a pre-recorded video file from your project directory.
        -   If the video appears sideways, use the "Fix video rotation" dropdown to correct it.
        -   Click "Analyze" to process the video and receive feedback.

---

## Project Structure

```
.
├── .venv/                     # Virtual environment folder
├── fuck.py                     # Main Streamlit application script
├── config.json                # Defines workout criteria and ideal angles
├── requirements.txt           # Python package dependencies
├── README.md                  # You are here
└── squats.mp4                 # Example video for testing mode (user-provided)
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
