import os
import requests
import time
import pickle
import streamlit as st
from googleapiclient.discovery import build
import plotly.graph_objects as go

# Load models and vectorizers
def load_models():
    with open("yt_ai_classifier_model_2.sav", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.sav", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

# Function to get video comments
def get_video_comments(video_id, youtube):
    comments = []
    results = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText',
        maxResults=100
    ).execute()

    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        if 'nextPageToken' in results:
            results = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                pageToken=results['nextPageToken'],
                maxResults=100
            ).execute()
        else:
            break

    return comments

# Function to classify comments
def classify_comments(comments, model, tfidf):
    categorized_comments = {'good': [], 'bad': [], 'neutral': []}

    for comment in comments:
        new_text = [comment]
        text_tfidf = tfidf.transform(new_text)
        model_output = model.predict(text_tfidf)
        output = model_output[0]

        if output == 2:
            categorized_comments['good'].append(comment)
        elif output == 0:
            categorized_comments['bad'].append(comment)
        else:
            categorized_comments['neutral'].append(comment)

    return categorized_comments

# Function to format comments as bullet points
def format_comments_as_bullets(comments):
    return "\n".join([f"- {comment}" for comment in comments])

# Function to plot interactive donut chart
def plot_interactive_donut_chart(categorized_comments):
    labels = ['Good', 'Bad', 'Neutral']
    sizes = [len(categorized_comments['good']), len(categorized_comments['bad']), len(categorized_comments['neutral'])]
    colors = ['#00ff00', '#ff0000', '#ffff00']

    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=sizes, 
        hole=0.4, 
        marker=dict(colors=colors), 
        hoverinfo='label+percent', 
        textinfo='value+percent', 
        textfont_size=15
    )])

    fig.update_layout(
        title_text='Sentiment Distribution of YouTube Comments',
        annotations=[dict(text='YCSA', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    st.plotly_chart(fig)

# Main function
def main():
    st.title("YouTube Live Chat and Comment Sentiment Analyzer")

    rainbow_line = """
    <hr style="height: 5px; border: none; background: linear-gradient(to right, 
        red, orange, yellow, green, blue, indigo, violet);">
    """
    st.markdown(rainbow_line, unsafe_allow_html=True)

    # Load models and vectorizers
    model, tfidf = load_models()

    # YouTube API setup
    API_KEY = "AIzaSyB-kZZzAsasrRK3OVOmg0id8cDiAx_wItE"
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    # Section for Live Chat Monitoring
    st.subheader("Live Chat Monitoring")
    video_url_input = st.text_input("Enter the YouTube video URL for live chat monitoring:")

    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False

    if st.button("Pause Chat Monitoring"):
        st.session_state.monitoring = False
        st.write("Chat monitoring paused. Click 'Start Chat Monitoring' to resume.")

    if st.button("Start Chat Monitoring"):
        if video_url_input:
            VIDEO_ID = video_url_input.split('v=')[1].split('&')[0] if 'v=' in video_url_input else video_url_input.split('/')[-1]

            video_url = "https://www.googleapis.com/youtube/v3/videos"
            video_params = {
                "part": "liveStreamingDetails",
                "id": VIDEO_ID,
                "key": API_KEY
            }
            response = requests.get(video_url, params=video_params).json()

            if "items" in response and response["items"]:
                live_chat_id = response["items"][0]["liveStreamingDetails"]["activeLiveChatId"]
                chat_url = "https://www.googleapis.com/youtube/v3/liveChat/messages"
                chat_params = {
                    "liveChatId": live_chat_id,
                    "part": "snippet,authorDetails",
                    "key": API_KEY
                }

                st.write("Monitoring live chat...")
                chat_placeholder = st.empty()
                all_messages = []
                
                st.session_state.monitoring = True

                while st.session_state.monitoring:
                    chat_response = requests.get(chat_url, params=chat_params).json()
                    if "items" in chat_response:
                        for item in chat_response["items"]:
                            author = item["authorDetails"]["displayName"]
