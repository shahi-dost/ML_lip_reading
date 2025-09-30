from unzip_untar import *
from track_on_video import *
import numpy as np
import pandas as pd
import gzip
import random
from Bio import AlignIO
from pathlib import Path

def read_align(path):
    # ChatGPT generated
    p = Path(path)
    def _open(p):
        with p.open("rb") as fh:
            head = fh.read(2)
        if p.suffix == ".gz" or head == b"\x1f\x8b":
            return gzip.open(p, "rt", encoding="utf-8", errors="replace")
        return p.open("r", encoding="utf-8", errors="replace")
    items = []
    with _open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue  # not enough columns
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                continue  # malformed numeric fields
            text = " ".join(parts[2:]) if len(parts) > 2 else ""
            items.append((start, end, text))
    return items

def create_df_from_video(video_path, normalize_by_width=False):
    # Shahi wrote this
    dict_of_points = track_lip_point_speeds(video_path, normalize_by_width=normalize_by_width)
    df_columns = ["frame", "time"]

    for i in range(len(dict_of_points[0]['points'])):
        df_columns.append(f"point_{i}_x")
        df_columns.append(f"point_{i}_y")
        df_columns.append(f"point_{i}_dx")
        df_columns.append(f"point_{i}_dy")
        df_columns.append(f"point_{i}_speed")

    df = pd.DataFrame(columns=df_columns)

    for i in range(len(dict_of_points)):
        df.loc[i, 'frame'] = dict_of_points[i]['frame']
        df.loc[i, 'time'] = dict_of_points[i]['time']
        for j in range(len(dict_of_points[i]['points'])):
            df.loc[i, f"point_{j}_x"] = dict_of_points[i]['points'][j][0]
            df.loc[i, f"point_{j}_y"] = dict_of_points[i]['points'][j][1]
            df.loc[i, f"point_{j}_dx"] = dict_of_points[i]['dx'][j]
            df.loc[i, f"point_{j}_dy"] = dict_of_points[i]['dy'][j]
            df.loc[i, f"point_{j}_speed"] = dict_of_points[i]['speed'][j]

    return df

def create_video_transcription_df_from_folder(video_folder_path, transcription_video_path, normalize_by_width=False):
    # Shahi wrote this
    participants = os.listdir(video_folder_path)
    participants = [i for i in participants if not i.startswith(".")]
    print(participants)
    
    return_df = pd.DataFrame()

    for i in participants:
        videos = os.listdir(os.path.join(video_folder_path, i))
        random.shuffle(videos)
        for j in videos[:5]:
            vid_df = create_df_from_video(os.path.join(video_folder_path, i, j), normalize_by_width=normalize_by_width)

            to_align = j.split(".")[0] + ".align"
            align = read_align(os.path.join(transcription_video_path, to_align))
            vid_df = vid_df.assign(participant=i, video=j, word='', word_begin_time=None, word_end_time=None)

            for row in vid_df.itertuples():
                for start, end, text in align:
                    start = start/25000
                    end = end/25000
                    if row.time >= start and row.time <= end:
                        vid_df.at[row.Index, 'word'] = text
                        vid_df.at[row.Index, 'word_begin_time'] = start
                        vid_df.at[row.Index, 'word_end_time'] = end
                        break

            return_df = pd.concat([return_df, vid_df.reindex(columns=vid_df.columns)],ignore_index=True)
    return return_df


def save_df_to_csv(df, out_path):
    df.to_csv(out_path, index=False)

# vd_path = "/Users/shahidullahdost/Documents/CS98/Word_Prediction/Data_Processing/Unzip_All_Video/"
# tr_path = "/Users/shahidullahdost/Documents/CS98/Word_Prediction/Data_Processing/Unzip_All_Transcriptions/align/"
# save_df = create_video_transcription_df_from_folder(vd_path, tr_path, normalize_by_width=False)
# out_path = "/Users/shahidullahdost/Documents/CS98/Word_Prediction/Data_Processing/processed_data.csv"
# save_df_to_csv(save_df, out_path)