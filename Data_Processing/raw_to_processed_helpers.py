from unzip_untar import *
from track_on_video import *
import numpy as np
import pandas as pd
import gzip
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


# frame_by_frame_df = pd.DataFrame(columns=["frame", "time", "centroid", "centroid_dx", "centroid_dy", "centroid_speed"])
# motion = track_lip_motion("/Users/shahidullahdost/Documents/CS98/Word_Prediction/Data_Processing/Unzip_All_Video/s1/bbaf2n.mpg")
# count = 0
# for r in motion:
#     print(r)
#     count +=1
#     if count == 3: break

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