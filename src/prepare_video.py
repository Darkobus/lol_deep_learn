from data_split import splitting_video_into_frames

VIDEO_PATH = r".\League of Legends_02-18-2026_22-6-43-0.mp4"
VIDEO_OUTPUT_PATH = r".\lol_deep_learn\lol\data\data3\images"

splitting_video_into_frames(VIDEO_PATH,VIDEO_OUTPUT_PATH)