from pytube import YouTube

videoURL = 'https://www.youtube.com/watch?v=BJyZCqAnVzc' #F1M2   2023 FMA District Bensalem Event
fileSaveDirectory = 'C:/Users/lukec/OneDrive/Documents/.Coding/Independent Study/Robotics Scouting/Match MP4s/'

YouTube(videoURL).streams.filter(file_extension='mp4', res='720p').first().download(fileSaveDirectory)
