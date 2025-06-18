import streamlit as st
import cv2
import mediapipe as mp
import speech_recognition as sr
import threading
import time
import os
import google.generativeai as genai
import queue
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import logging
from pathlib import Path
import json
import pandas as pd
import requests # The correct library for direct API calls

# --- Configuration ---
KEYWORD = "start workout"
RECORDING_DURATION = 15
GEMINI_MODEL = "gemini-1.5-pro-latest"
WHISPER_MODEL = "openai/whisper-base.en"
VOICE_ID_PHIL = "3mB9h1SrmasYAyvcVvMn"
ELEVENLABS_API_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID_PHIL}"
GREEN_ZONE_PERCENTAGE = 0.5

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Key Setups ---
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    st.error("🚨 GOOGLE_API_KEY environment variable not set! Please set it before running.")
    st.stop()

try:
    ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
except KeyError:
    st.error("🚨 ELEVENLABS_API_KEY environment variable not set! Please set it before running.")
    st.stop()

# --- MediaPipe Pose Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_for_video = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
pose_for_live = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- State Management ---
if "status" not in st.session_state: st.session_state.status = "waiting"
if "listener_thread" not in st.session_state: st.session_state.listener_thread = None
if "frames" not in st.session_state: st.session_state.frames = []
if "start_time" not in st.session_state: st.session_state.start_time = 0
if "keyword_q" not in st.session_state: st.session_state.keyword_q = queue.Queue()
if "voice_enabled" not in st.session_state: st.session_state.voice_enabled = True
if "selected_workout" not in st.session_state: st.session_state.selected_workout = "Squat"
if "feedback_text" not in st.session_state: st.session_state.feedback_text = None
if "feedback_audio_path" not in st.session_state: st.session_state.feedback_audio_path = None
if "testing_mode" not in st.session_state: st.session_state.testing_mode = False
if "test_video_path" not in st.session_state: st.session_state.test_video_path = None
if "performance_data" not in st.session_state: st.session_state.performance_data = {}

# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_whisper_model():
    st.info(f"Loading Whisper model '{WHISPER_MODEL}'...")
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL)
    st.success("Whisper model loaded successfully.")
    return processor, model

@st.cache_data
def find_video_files():
    video_files = [f.name for f in Path(".").glob("*.mp4")]
    video_files.extend([f.name for f in Path(".").glob("*.mov")])
    return video_files

@st.cache_data
def load_workout_configs(filepath="config.json"):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"FATAL ERROR: `config.json` not found. Please create it."); return None
    except json.JSONDecodeError:
        st.error(f"FATAL ERROR: Could not decode `config.json`. Check syntax."); return None

### --- FINAL TTS FUNCTION (using requests) --- ###
def text_to_speech(text):
    """Converts text to speech using a direct API call to ElevenLabs and saves the audio file."""
    filepath = "feedback.mp3"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    data = {"text": text, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
    
    try:
        response = requests.post(ELEVENLABS_API_URL, json=data, headers=headers, timeout=60)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return filepath
    except requests.exceptions.RequestException as e:
        logging.error("CRITICAL: ElevenLabs API request failed.", exc_info=True)
        st.error(f"Failed to generate audio with ElevenLabs. Check API key, account status, and internet connection. Error: {e}")
        return None

def get_angle_color(deviation, threshold):
    if abs(deviation) <= threshold * GREEN_ZONE_PERCENTAGE: return (0, 255, 0)
    elif abs(deviation) <= threshold: return (0, 255, 255)
    else: return (0, 0, 255)

def style_deviation(val, threshold):
    if abs(val) <= threshold * GREEN_ZONE_PERCENTAGE: color = 'green'
    elif abs(val) <= threshold: color = 'orange'
    else: color = 'red'
    return f'color: {color}'

def calculate_angle(p1, p2, p3):
    p1 = np.array([p1.x, p1.y]); p2 = np.array([p2.x, p2.y]); p3 = np.array([p3.x, p3.y])
    vec1 = p1 - p2; vec2 = p3 - p2
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)); return np.degrees(angle)

def analyze_performance_data(performance_data, workout_config):
    summary = "Biomechanical Analysis Results:\n"
    for criterion in workout_config['criteria']:
        name = criterion['name']; angles = performance_data.get(name)
        if not angles: summary += f"- {name}: No data collected.\n"; continue
        avg_angle = np.mean(angles); min_angle = np.min(angles); max_angle = np.max(angles); ideal = criterion['ideal_angle']
        summary += f"- **{name}**:\n  - Ideal: {ideal}°\n  - Your Average: {avg_angle:.1f}°\n  - Your Range: {min_angle:.1f}° to {max_angle:.1f}°\n"
        focus_metric = min_angle if criterion.get('focus') == 'min' else avg_angle
        deviation = focus_metric - ideal
        if abs(deviation) > criterion['threshold']:
            summary += f"  - **Deviation Note**: Your angle was {'significantly smaller (tighter)' if deviation < 0 else 'significantly larger (wider)'} than the ideal.\n"
        else:
             summary += f"  - **Deviation Note**: Your form was within the acceptable range for this metric.\n"
    return summary

def check_trigger_condition(results, workout_config, landmark_map):
    if "trigger" not in workout_config: return True
    trigger_cfg = workout_config["trigger"]; trigger_metric_name = trigger_cfg["metric_name"]
    trigger_criterion = next((c for c in workout_config['criteria'] if c['name'] == trigger_metric_name), None)
    if not trigger_criterion: return True
    try:
        p1 = results.pose_landmarks.landmark[landmark_map[trigger_criterion['landmarks'][0]]]; p2 = results.pose_landmarks.landmark[landmark_map[trigger_criterion['landmarks'][1]]]; p3 = results.pose_landmarks.landmark[landmark_map[trigger_criterion['landmarks'][2]]]
        current_angle = calculate_angle(p1, p2, p3)
        if trigger_cfg["comparison"] == "less_than": return current_angle < trigger_cfg["threshold"]
        elif trigger_cfg["comparison"] == "greater_than": return current_angle > trigger_cfg["threshold"]
        return False
    except: return False

def analyze_recording(video_path, workout_type, performance_summary):
    st.info("Uploading video for analysis...")
    try:
        video_file = genai.upload_file(path=video_path)
        while video_file.state.name == "PROCESSING": time.sleep(2); video_file = genai.get_file(video_file.name)
        if video_file.state.name == "FAILED": st.error("Video processing failed."); return "Sorry, I couldn't process the video file."
        st.info("Video processed. Asking Gemini for hyper-specific feedback...")
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        prompt = f"""You are an elite AI fitness coach. Your feedback must be direct, data-driven, and based *only* on the quantitative analysis provided. The user performed a '{workout_type}'. I have analyzed their form and a video is attached for visual context. **DO NOT give generic advice.** Use the following biomechanical data to give 2-3 hyper-specific, actionable tips. {performance_summary} Based *strictly* on the 'Deviation Notes' in the data above, generate your feedback. Start with an encouraging sentence, then list the specific corrections. For example, if the data says "Left Knee Bend... angle was significantly larger", your feedback should be "Great effort! I noticed your squat might have been a bit shallow, as your knee angle was larger than the ideal 90 degrees. Try to sink your hips lower on the next set." Keep the entire response under 4 sentences."""
        response = model.generate_content([prompt, video_file], request_options={"timeout": 120})
        genai.delete_file(video_file.name); return response.text
    except Exception as e:
        logging.error("--- GEMINI ANALYSIS FAILED ---", exc_info=True); st.error(f"An error occurred during analysis. Check console for details. Error: {e}"); return "I encountered an error while analyzing your workout."

def listen_for_keyword(q, processor, model):
    r = sr.Recognizer(); mic = sr.Microphone(sample_rate=16000)
    with mic as source: r.adjust_for_ambient_noise(source, duration=0.5)
    print("Listener thread started. Say the keyword.")
    while True:
        try:
            with mic as source: audio = r.listen(source, phrase_time_limit=4)
            raw_data = audio.get_raw_data(); sample_rate = audio.sample_rate
            audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            input_features = processor(audio_np, sampling_rate=sample_rate, return_tensors="pt").input_features
            predicted_ids = model.generate(input_features, max_new_tokens=100)
            transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower().strip()
            print(f"Whisper heard: {transcript}")
            if KEYWORD in transcript: print("Keyword detected! Signaling main thread.");
            while not q.empty(): q.get(); q.put("KEYWORD_DETECTED")
        except sr.UnknownValueError: pass
        except Exception as e: print(f"An error occurred in the listener thread: {e}"); time.sleep(1)

def main():
    st.set_page_config(layout="wide"); st.title("Data-Driven AI Fitness Coach")
    workout_configs = load_workout_configs()
    whisper_processor, whisper_model = load_whisper_model()
    if not workout_configs: st.stop()
    if 'selected_workout' not in st.session_state or st.session_state.selected_workout not in workout_configs:
        st.session_state.selected_workout = list(workout_configs.keys())[0]
    landmark_map = {name.name: i for i, name in enumerate(mp_pose.PoseLandmark)}
    c1, c2, c3 = st.columns([1.5, 1, 2]);
    with c1: st.session_state.testing_mode = st.toggle("Testing Mode", value=st.session_state.testing_mode)
    if not st.session_state.testing_mode:
        with c2: st.session_state.voice_enabled = st.toggle("Voice Commands", value=st.session_state.voice_enabled)
    with c3: st.session_state.selected_workout = st.selectbox("Workout Type:", list(workout_configs.keys()), index=list(workout_configs.keys()).index(st.session_state.selected_workout))
    st.divider()
    if st.session_state.status == "feedback":
        st.success("Analysis Complete!"); st.markdown(f"**Coach's Feedback for your {st.session_state.selected_workout}:**"); st.markdown(st.session_state.feedback_text, unsafe_allow_html=True)
        if st.session_state.feedback_audio_path: st.audio(st.session_state.feedback_audio_path, autoplay=True)
        if st.button("Start Next Set"):
            if st.session_state.feedback_audio_path and os.path.exists(st.session_state.feedback_audio_path): os.remove(st.session_state.feedback_audio_path)
            st.session_state.status = "waiting"; st.session_state.listener_thread = None; st.session_state.frames = []; st.session_state.feedback_text = None; st.session_state.feedback_audio_path = None; st.session_state.performance_data = {}
            st.rerun()
        st.stop()
    if st.session_state.status == "analyzing":
        with st.spinner("Set complete! Analyzing your form..."):
            video_path = "workout_capture.mp4";
            if st.session_state.frames:
                workout_config = workout_configs[st.session_state.selected_workout]; performance_summary = analyze_performance_data(st.session_state.performance_data, workout_config)
                height, width, _ = st.session_state.frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v'); out = cv2.VideoWriter(video_path, fourcc, 15.0, (width, height))
                for frame in st.session_state.frames: out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                feedback_text = analyze_recording(video_path, st.session_state.selected_workout, performance_summary)
                audio_path = text_to_speech(feedback_text)
                os.remove(video_path)
                st.session_state.feedback_text = feedback_text; st.session_state.feedback_audio_path = audio_path; st.session_state.status = "feedback"
                st.rerun()
            else: st.error("No frames were processed."); st.session_state.status = "waiting"; st.rerun()
        st.stop()
    video_col, data_col = st.columns([2, 1]); frame_placeholder = video_col.empty(); data_placeholder = data_col.empty()
    if st.session_state.testing_mode:
        video_files = find_video_files();
        if not video_files: st.error("No video files found."); st.stop()
        col1, col2 = st.columns(2);
        with col1: st.session_state.test_video_path = st.selectbox("Select a test video:", video_files)
        with col2: rotation_option = st.selectbox("Fix video rotation:", ['None', 'Rotate 90° Counter-Clockwise', 'Rotate 90° Clockwise', 'Rotate 180°'])
        if st.button(f"Analyze {st.session_state.test_video_path}"): st.session_state.status = "processing_test_video"; st.rerun()
        if st.session_state.status == "processing_test_video":
            st.session_state.frames = []; st.session_state.performance_data = {c['name']: [] for c in workout_configs[st.session_state.selected_workout]['criteria']}
            cap = cv2.VideoCapture(st.session_state.test_video_path)
            with st.spinner(f"Processing '{st.session_state.test_video_path}'..."):
                while cap.isOpened():
                    ret, frame = cap.read();
                    if not ret: break
                    if rotation_option == 'Rotate 90° Counter-Clockwise': frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rotation_option == 'Rotate 90° Clockwise': frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation_option == 'Rotate 180°': frame = cv2.rotate(frame, cv2.ROTATE_180)
                    h, w, _ = frame.shape; rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); results = pose_for_video.process(rgb_frame)
                    df_data = []
                    if results.pose_landmarks:
                        workout_config = workout_configs[st.session_state.selected_workout]; is_in_pose = check_trigger_condition(results, workout_config, landmark_map)
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        for criterion in workout_config['criteria']:
                            ideal_angle = criterion['ideal_angle']; current_angle = 0; deviation = 0; color = (255, 255, 255)
                            try:
                                p1 = results.pose_landmarks.landmark[landmark_map[criterion['landmarks'][0]]]; p2 = results.pose_landmarks.landmark[landmark_map[criterion['landmarks'][1]]]; p3 = results.pose_landmarks.landmark[landmark_map[criterion['landmarks'][2]]]
                                current_angle = calculate_angle(p1, p2, p3); deviation = current_angle - ideal_angle
                                if is_in_pose: st.session_state.performance_data[criterion['name']].append(current_angle); color = get_angle_color(deviation, criterion['threshold'])
                                cv2.putText(frame, f"{current_angle:.1f}", (int(p2.x * w), int(p2.y * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            except: pass
                            df_data.append({"Metric": criterion['name'], "Ideal": ideal_angle, "Actual": f"{current_angle:.1f}", "Deviation": deviation if is_in_pose else 0})
                    if df_data:
                        df = pd.DataFrame(df_data)
                        if not df.empty: styled_df = df.style.apply(lambda row: [style_deviation(row['Deviation'], workout_config['criteria'][i]['threshold']) for i in range(len(row))], axis=1, subset=['Deviation']).format({'Deviation': '{:+.1f}'}); data_placeholder.dataframe(styled_df, use_container_width=True)
                    st.session_state.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); frame_placeholder.image(frame, channels="BGR")
            cap.release(); st.session_state.status = "analyzing"; st.rerun()
    else:
        if st.session_state.status == "waiting":
            if video_col.button(f"Start {st.session_state.selected_workout} Set Manually"): st.session_state.status = "countdown"; st.rerun()
        cap = cv2.VideoCapture(0)
        try:
            if not cap.isOpened(): st.error("Cannot open camera."); st.stop()
            if st.session_state.voice_enabled and st.session_state.listener_thread is None:
                thread = threading.Thread(target=listen_for_keyword, args=(st.session_state.keyword_q, whisper_processor, whisper_model), daemon=True); st.session_state.listener_thread = thread; thread.start()
            while True:
                if not st.session_state.keyword_q.empty():
                    if st.session_state.keyword_q.get() == "KEYWORD_DETECTED" and st.session_state.status == 'waiting': st.session_state.status = "countdown"; st.rerun()
                if st.session_state.status not in ["waiting", "countdown", "recording"]: break
                ret, frame = cap.read();
                if not ret: st.error("Failed to grab frame."); break
                frame = cv2.flip(frame, 1); h, w, _ = frame.shape; rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); results = pose_for_live.process(rgb_frame)
                if results.pose_landmarks: mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if st.session_state.status == "waiting":
                    cv2.putText(frame, "STATUS: WAITING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2); data_placeholder.info("Ready to begin. The data table will appear here.")
                elif st.session_state.status == "countdown":
                    st.session_state.performance_data = {c['name']: [] for c in workout_configs[st.session_state.selected_workout]['criteria']}
                    for i in range(3, 0, -1):
                        countdown_frame = frame.copy(); cv2.putText(countdown_frame, str(i), (int(w/2) - 50, int(h/2) + 50), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10); frame_placeholder.image(countdown_frame, channels="BGR"); time.sleep(1)
                    st.session_state.start_time = time.time(); st.session_state.status = "recording"; st.rerun()
                elif st.session_state.status == "recording":
                    elapsed_time = time.time() - st.session_state.start_time; countdown = max(0, RECORDING_DURATION - int(elapsed_time))
                    cv2.putText(frame, f"{st.session_state.selected_workout}: {countdown}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    df_data = []
                    if results.pose_landmarks:
                        workout_config = workout_configs[st.session_state.selected_workout]; is_in_pose = check_trigger_condition(results, workout_config, landmark_map)
                        for criterion in workout_config['criteria']:
                            ideal_angle = criterion['ideal_angle']; current_angle = 0; deviation = 0; color = (255, 255, 255)
                            try:
                                p1 = results.pose_landmarks.landmark[landmark_map[criterion['landmarks'][0]]]; p2 = results.pose_landmarks.landmark[landmark_map[criterion['landmarks'][1]]]; p3 = results.pose_landmarks.landmark[landmark_map[criterion['landmarks'][2]]]
                                current_angle = calculate_angle(p1, p2, p3); deviation = current_angle - ideal_angle
                                if is_in_pose: st.session_state.performance_data[criterion['name']].append(current_angle); color = get_angle_color(deviation, criterion['threshold'])
                                cv2.putText(frame, f"{current_angle:.1f}", (int(p2.x * w), int(p2.y * h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            except: pass
                            df_data.append({"Metric": criterion['name'], "Ideal": ideal_angle, "Actual": f"{current_angle:.1f}", "Deviation": deviation if is_in_pose else 0})
                    if df_data:
                        df = pd.DataFrame(df_data)
                        if not df.empty:
                            styled_df = df.style.apply(lambda row: [style_deviation(row['Deviation'], workout_config['criteria'][i]['threshold']) for i in range(len(row))], axis=1, subset=['Deviation']).format({'Deviation': '{:+.1f}'})
                            data_placeholder.dataframe(styled_df, use_container_width=True)
                    else: data_placeholder.info("Waiting for pose detection...")
                    st.session_state.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if elapsed_time >= RECORDING_DURATION: st.session_state.status = "analyzing"
                frame_placeholder.image(frame, channels="BGR")
        finally:
            if cap.isOpened(): cap.release(); print("Camera released.")

if __name__ == "__main__":
    main()