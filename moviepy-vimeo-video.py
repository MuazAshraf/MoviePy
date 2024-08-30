from flask import Flask, request, jsonify, send_file
import requests
import os
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, TextClip, ImageClip, CompositeVideoClip, vfx, clips_array, concatenate_videoclips
from moviepy.config import change_settings  
import time
import cv2
from skimage.metrics import structural_similarity as ssim
import threading, uuid, traceback


app = Flask(__name__)
# Set ImageMagick path
change_settings({"IMAGEMAGICK_BINARY": "C:\\Path\\To\\ImageMagick\\magick.exe"})

# Vimeo access token (embedded)
access_token = 'token'
# Directory where videos will be saved and processed
DOWNLOAD_FOLDER = 'downloaded_videos'
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


tasks_status = {}
# downloads a file from a given URL and saves it locally
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded file from {url} to {output_path}")
        return True
    print(f"Failed to download file from {url}")
    return False

# calculates the position where an overlay (like text or logo) should be placed on the video, based on the provided position (e.g., 'top_left', 'center').
def calculate_position(position, clip_size, video_size, padding=20):
    if position == 'top_left':
        return (padding, padding)
    elif position == 'top_right':
        return (video_size[0] - clip_size[0] - padding, padding)
    elif position == 'bottom_left':
        return (padding, video_size[1] - clip_size[1] - padding)
    elif position == 'bottom_right':
        return (video_size[0] - clip_size[0] - padding, video_size[1] - clip_size[1] - padding)
    elif position == 'center':
        return ((video_size[0] - clip_size[0]) // 2, (video_size[1] - clip_size[1]) // 2)
    else:
        return (0, 0)
#detect scene on the basis of similarity score 
def detect_scenes(video_path):
    cap = cv2.VideoCapture(video_path)
    scenes = []
    prev_frame = None
    frame_count = 0

    print("Starting scene detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            score = ssim(prev_frame, gray_frame)
            if score < 0.5:  # Threshold to detect scene change
                scenes.append(frame_count)
                print(f"Scene change detected at frame {frame_count}, score: {score}")

        prev_frame = gray_frame
        frame_count += 1

    cap.release()
    print(f"Scene detection completed. {len(scenes)} scenes detected.")
    return scenes



def process_video(video_path, processing_instructions, task_id):
    try:
        video_clip = VideoFileClip(video_path)
        scenes = detect_scenes(video_path)

        if not scenes:
            print("No scenes detected. Using the entire video.")
            important_clips = [video_clip]
        else:
            important_clips = []
            for i in range(len(scenes) - 1):
                start_time = scenes[i] / video_clip.fps
                end_time = scenes[i + 1] / video_clip.fps
                important_clips.append(video_clip.subclip(start_time, end_time))
                print(f"Extracted clip from {start_time} to {end_time} seconds.")

        combined_clip = concatenate_videoclips(important_clips)
        clips = []

        for task in processing_instructions:
            if task['action'] == 'trim':
                start_time = task.get('start', 0)
                end_time = task.get('end', video_clip.duration)
                print(f"Trimming video from {start_time} to {end_time} seconds.")
                combined_clip = combined_clip.subclip(start_time, end_time)

            elif task['action'] == 'crop':
                target_aspect_ratio = task.get('aspect_ratio', [9, 16])
                print(f"Cropping video to aspect ratio: {target_aspect_ratio}")

                resized_clip = combined_clip.resize(width=target_aspect_ratio[0])
                final_clip = clips_array([[resized_clip]])
                final_clip_resized = final_clip.fx(vfx.resize, target_aspect_ratio)
                combined_clip = final_clip_resized

            elif task['action'] == 'add_text':
                text_clip = TextClip(
                    task['text'],
                    fontsize=task.get('fontsize', 70),
                    color=task.get('font_color', 'white'),
                    font='Arial-Bold',
                    bg_color=task.get('background_color', None)
                ).set_duration(combined_clip.duration)

                position = calculate_position(task.get('position'), text_clip.size, combined_clip.size)
                text_clip = text_clip.set_position(position)

                if task.get('effect') == 'fadein':
                    text_clip = text_clip.fadein(1)

                if 'opacity' in task:
                    text_clip = text_clip.set_opacity(task.get('opacity', 1.0))

                clips.append(text_clip)
                print(f"Added text: '{task['text']}' at position {task['position']}")

            elif task['action'] == 'add_logo':
                logo_url = task.get('logo_url')
                logo_path = os.path.join(DOWNLOAD_FOLDER, 'logo.png')
                if not download_file(logo_url, logo_path):
                    raise Exception("Failed to download logo from the provided URL")

                logo_clip = ImageClip(logo_path).set_duration(combined_clip.duration)
                logo_clip = logo_clip.resize(task.get('size', (100, 100)))

                position = calculate_position(task.get('position'), logo_clip.size, combined_clip.size)
                logo_clip = logo_clip.set_position(position).set_opacity(task.get('opacity', 0.7))

                clips.append(logo_clip)
                print(f"Added logo from {logo_url} at position {task['position']}")

            elif task['action'] == 'add_background_audio':
                audio_url = task.get('audio_url')
                audio_path = os.path.join(DOWNLOAD_FOLDER, 'background_audio.mp3')
                if not download_file(audio_url, audio_path):
                    raise Exception("Failed to download audio from the provided URL")

                background_music = AudioFileClip(audio_path)
                background_music = background_music.volumex(0.3)
                background_music = background_music.subclip(0, min(combined_clip.duration, background_music.duration - 0.05))
                combined_audio = CompositeAudioClip([combined_clip.audio, background_music])
                combined_clip = combined_clip.set_audio(combined_audio)
                print(f"Added background audio from {audio_url}")

        # Composite all video clips (including the original video, text, and logo)
        composite_video = CompositeVideoClip([combined_clip] + clips)

        # Save the final video with all edits applied
        output_path = os.path.join(DOWNLOAD_FOLDER, 'processed_video.mp4')
        composite_video.write_videofile(output_path, codec="libx264", fps=30)
        print(f"Final video saved to {output_path}")

        # Update status to 'Completed'
        tasks_status[task_id] = 'Completed'
        return output_path

    except Exception as e:
        print(f"Error processing video: {e}")
        # Update status to 'Failed'
        tasks_status[task_id] = 'Failed'
        raise e

def upload_to_vimeo(video_path):
    video_size = os.path.getsize(video_path)
    video_name = os.path.basename(video_path)
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.vimeo.*+json;version=3.4',
    }

    body = {
        "upload": {
            "approach": "tus",
            "size": str(video_size)
        }
    }

    response = requests.post('https://api.vimeo.com/me/videos', headers=headers, json=body)
    if not response.ok:
        return None, f"Failed to create upload session. Status Code: {response.status_code}, Response: {response.text}"

    upload_link = response.json().get('upload', {}).get('upload_link', '')
    video_uri = response.json().get('uri', '')  # This is the video URI needed for the final URL
    if not upload_link:
        return None, "Upload link not found in response."

    chunk_size = 1048576  # 1MB chunks
    offset = 0
    retries = 0
    max_retries = 5

    with open(video_path, 'rb') as f:
        while offset < video_size:
            f.seek(offset)
            data = f.read(chunk_size)
            patch_headers = {
                'Authorization': f'Bearer {access_token}',
                'Tus-Resumable': '1.0.0',
                'Upload-Offset': str(offset),
                'Content-Type': 'application/offset+octet-stream',
            }
            response = requests.patch(upload_link, headers=patch_headers, data=data)
            if response.ok:
                offset = int(response.headers['Upload-Offset'])
                retries = 0  # Reset retries after a successful chunk upload
            else:
                if response.status_code == 404:
                    return None, "Upload link is not found, stopping the upload."
                retries += 1
                if retries >= max_retries:
                    return None, "Maximum retries reached for chunk upload. Upload failed."
                time.sleep(5)  # Wait before retrying

    if offset == video_size:
        # The upload is complete, now we need to get the video details
        video_details_response = requests.get(f'https://api.vimeo.com{video_uri}', headers=headers)
        if video_details_response.ok:
            video_link = video_details_response.json().get('link', '')
            if video_link:
                return video_link, None  # Return Vimeo link and no error
            else:
                return None, "Failed to retrieve final video link."
        else:
            return None, f"Failed to retrieve video details. Status Code: {video_details_response.status_code}"

    return None, "Upload failed during chunk upload."

@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    video_url = request.json.get('video_url')
    processing_instructions = request.json.get('processing_instructions')

    if not video_url or not processing_instructions:
        return jsonify({'error': 'video_url and processing_instructions are required'}), 400

    video_path = os.path.join(DOWNLOAD_FOLDER, 'input_video.mp4')
    task_id = str(uuid.uuid4())

    if not download_file(video_url, video_path):
        return jsonify({'error': 'Failed to download video'}), 500

    # Initialize status to 'Processing'
    tasks_status[task_id] = 'Processing'

    # Start video processing in a separate thread
    def process_video_thread():
        try:
            print("Starting video processing thread.")
            
            # Process the video
            output_path = process_video(video_path, processing_instructions, task_id)
            if not output_path:
                raise Exception("Video processing failed, no output path returned.")
            print(f"Processed video saved to: {output_path}")
            
            # Upload the processed video
            upload_link, error = upload_to_vimeo(output_path)
            if error:
                print(f"Upload error: {error}")
                tasks_status[task_id] = 'Failed'
                return
            
            print(f"Upload successful. Vimeo URL: {upload_link}")
            tasks_status[task_id] = 'Completed'
        except Exception as e:
            # Log the exception and traceback
            error_message = f"Exception in video processing thread: {str(e)}"
            print(error_message)
            traceback.print_exc()
            tasks_status[task_id] = 'Failed'

    threading.Thread(target=process_video_thread).start()

    return jsonify({'message': 'Video is being processed. You will receive a notification once it is ready.', 'task_id': task_id}), 202

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    status = tasks_status.get(task_id, 'Not found')
    return jsonify({'task_id': task_id, 'status': status})

if __name__ == '__main__':
    app.run(debug=True)
