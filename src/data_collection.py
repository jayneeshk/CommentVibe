from googleapiclient.discovery import build
import pandas as pd
import json
import time
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('YOUTUBE_API_KEY')
# API_KEY = 'api_key'
youtube = build('youtube', 'v3', developerKey=api_key)

def get_video_comments(video_id, max_results=100):
    comments = []
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=max_results,
        textFormat='plainText'
    ).execute()

    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        # Check if there is a next page token to retrieve more comments
        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=max_results,
                textFormat='plainText'
            ).execute()
        else:
            break

    return comments

video_ids = ['cXxmbemS6XM']  # Replace with actual video IDs

all_comments = []

for video_id in video_ids:
    comments = get_video_comments(video_id)
    all_comments.extend(comments)
    time.sleep(1)  # To avoid hitting API rate limits

# Save the collected comments to a CSV file
df = pd.DataFrame(all_comments, columns=['comment'])
df.to_csv('data/raw/youtube_comments.csv', index=False)
