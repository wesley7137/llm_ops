# Importing required libraries
from youtube_transcript_api import YouTubeTranscriptApi

# Function to download and save transcript
def download_transcript(video_id, filename):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        with open(filename, 'w') as f:
            buffer_text = ""
            for i, entry in enumerate(transcript):
                buffer_text += entry['text'] + " "
                if (i + 1) % 4 == 0:  # Concatenate every 4 lines
                    f.write(f"{buffer_text.strip()}\n")
                    buffer_text = ""
            if buffer_text:  # Write any remaining text
                f.write(f"{buffer_text.strip()}\n")
        print(f"Transcript saved to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main function
if __name__ == "__main__":
    youtube_url = input("Enter the YouTube video URL: ")
    video_id = youtube_url.split("v=")[1]
    filename = f"{video_id}_transcript.txt"
    download_transcript(video_id, filename)

