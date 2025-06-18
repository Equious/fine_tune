#!/usr/bin/env python3
import subprocess
import os
import shutil

def check_ffmpeg():
    """Checks if ffmpeg is installed and in the system's PATH."""
    if shutil.which("ffmpeg") is None:
        print("❌ ERROR: FFmpeg not found. Please install FFmpeg.")
        print("On macOS with Homebrew, run: brew install ffmpeg")
        return False
    return True

def cut_audio_to_size(input_audio_path, output_audio_path, target_size_mb, scan_duration_seconds=300):
    """
    Creates an audio segment from the start of the input audio file that is
    approximately target_size_mb.

    It processes up to 'scan_duration_seconds' of the input audio and stops
    when the output file size limit is reached.

    Args:
        input_audio_path (str): Path to the input audio file (e.g., .mp3, .m4a).
        output_audio_path (str): Path to save the cut audio file.
        target_size_mb (float): The target output file size in Megabytes.
        scan_duration_seconds (int): The maximum duration from the input to process.
    """
    if not check_ffmpeg():
        return

    if not os.path.exists(input_audio_path):
        print(f"❌ Error: Input audio file not found at '{input_audio_path}'")
        return

    target_size_bytes = int(target_size_mb * 1024 * 1024)  # Convert MB to bytes

    # --- The FFmpeg command, adapted for audio ---
    # -i : input file
    # -t : duration of the input to process (an upper limit for the cut)
    # -fs : limit the output file size (in bytes). This is the key flag.
    # -c:a copy : Copy the audio stream directly without re-encoding.
    #             This is extremely fast and preserves the original quality.
    # -y : overwrite output file if it exists
    command = [
        'ffmpeg',
        '-i', input_audio_path,
        '-t', str(scan_duration_seconds),
        '-fs', str(target_size_bytes),
        '-c:a', 'copy',  # Use stream copy for speed and quality
        '-y',
        output_audio_path
    ]

    print(f"▶️  Attempting to cut '{os.path.basename(input_audio_path)}' to approx {target_size_mb}MB.")
    print(f"    (Processing up to the first {scan_duration_seconds} seconds of input)")
    # print(f"FFmpeg command: {' '.join(command)}") # Uncomment for debugging

    try:
        # Using DEVNULL to keep the console clean from ffmpeg's verbose output
        process = subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        print(f"\n✅ Audio processing complete. Output saved to: '{output_audio_path}'")
        
        actual_size_bytes = os.path.getsize(output_audio_path)
        actual_size_mb = actual_size_bytes / (1024 * 1024)
        print(f"   Actual output file size: {actual_size_mb:.2f} MB")
        
        # Your excellent warning logic, slightly tweaked for clarity
        if actual_size_mb < 0.01:
            print("⚠️  Warning: Output file is nearly 0 bytes. This might mean 'scan_duration_seconds' was too short or the target size was too small for any output.")
        elif actual_size_mb < target_size_mb * 0.9:
            print(f"⚠️  Warning: Actual size ({actual_size_mb:.2f}MB) is smaller than target ({target_size_mb}MB).")
            print("   This likely means the entire scanned duration was encoded and was still smaller than the target size.")
            print("   Consider increasing 'scan_duration_seconds' if the source file is longer.")

    except subprocess.CalledProcessError as e:
        print("\n❌ Error during FFmpeg processing:")
        # Stderr often contains the useful error message from ffmpeg
        print(e.stderr.decode(errors='ignore'))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # --- IMPORTANT: CHANGE THIS TO THE PATH OF YOUR MP3 FILE ---
    # Example for a file in your user's Music folder on a Mac:
    input_f = os.path.expanduser("/Users/luna_remote/workout/patrick_audio.mp3")
    
    # Check if the example file exists before running
    if not os.path.exists(input_f):
        print(f"File not found: {input_f}")
        print("Please edit the script and set the 'input_f' variable to the correct path of your audio file.")
    else:
        # Define the output file and target size
        base, ext = os.path.splitext(input_f)
        output_f = f"{base}_8MB_cut{ext}"
        target_mb = 8.0
        
        # How many seconds of the source audio to look at.
        # If your audio has a low bitrate, 8MB might be longer than 300 seconds.
        # If it's a high-bitrate file, 8MB might be reached in under a minute.
        scan_duration = 300 # Process the first 5 minutes of input at most.

        cut_audio_to_size(input_f, output_f, target_mb, scan_duration)