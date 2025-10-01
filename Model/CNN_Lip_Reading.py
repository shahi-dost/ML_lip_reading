'''
 ---- Code Author ---
 Author for lines [1:205] > Jenny Lee 
 Author for lines [206:] > 
 
 Short dscptn: this is a CNN-GRU model using Attention for our lip reading feature sequence classification

 - Base architecture of the model: Used/followed general concepts from RNN/GRU tutorials. Got the idea from chatgpt to use them 
 and then followed this tutorial: https://www.youtube.com/watch?v=0_PgWWmauHk
 - Complex/New Special Features I never used before: model attention and sample weight needed a lot of tutorials + model support 
  - Sample weighting:https://stackoverflow.com/questions/48315094/using-sample-weight-in-keras-for-sequence-labelling 
  - Modeel attention I needed for keras guides
    https://machinelearningmastery.com/adding-a-custom-attention-layer-to-recurrent-neural-network-in-keras/
    https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
 - Data prep/preprocessing: We studied time-series data pre-processing guides to do this
 here: https://discuss.pytorch.org/t/pytorch-dataloader-with-variable-sequence-lengths-inputs/98425
 - DEBUGGING: We used chat in several places so we could quickly move forward to a working product. But our prompts consisted of 'teach me how to debug this' (got lots of help with tensor shape problems esp in attention lyrs)
'''

import os, csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# CHANGE DATA PATH AS NECESSARY
DATA_PATH = "/Users/jenniferlee/Documents/CS98/hack-a-thing-shahijennifer/Processed_Data/processed_data.csv"
T = 75
H1, H2 = 128, 64
SLOTS = 6
SEED = 42

def set_seed(seed=SEED):
    # setting so numpy and tensorflow use the same seed so we can reproduce whatever accuracy we get
    import random
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def collapsed_words(g):
    # collapses the repeating words + removes the 'sil' in the data
    seq = g["word"].astype(str).tolist()
    out, prev = [], None
    for w in seq:
        if w == prev:
            continue
        prev = w
        if w != "sil":
            out.append(w)
    return out

def fix_length(X_2d, T=75):
    # Got this logic from time-series preprocessing tutorials, studied custom padding/resampling
    t, F = X_2d.shape
    if t == T:
        return X_2d.astype(np.float32)
    if t > T:
        # downsampling to fix length T (did this bc ran into errors)
        idx = np.linspace(0, t-1, T).round().astype(int)
        return X_2d[idx].astype(np.float32)
    # padding shorter sequences by repeating the last frame >not ideal but data is small and not the most consistent so
    pad = np.repeat(X_2d[-1:, :], T - t, axis=0)
    return np.concatenate([X_2d, pad], axis=0).astype(np.float32)

def finite_or_die(**arrays):
    # checking for NaN or infinit vals (suggested by CHAT!!)
    for name, arr in arrays.items():
        if not np.isfinite(arr).all():
            bad = np.argwhere(~np.isfinite(arr))
            raise ValueError(f"[{name}] has infinite val here {bad[:5]}")

set_seed()
df = pd.read_csv(DATA_PATH)
assert "video" in df.columns and "frame" in df.columns and "word" in df.columns
feat_cols = [c for c in df.columns if c.startswith("point_")]
assert len(feat_cols) == 200

words = sorted(df["word"].astype(str).unique().tolist())
word2id = {w:i for i,w in enumerate(words)}
id2word = {i:w for w,i in word2id.items()}
num_classes = len(word2id)

# splitting data by vid ID to prevent data spreading 
videos = df["video"].dropna().unique()
rng = np.random.default_rng(SEED); rng.shuffle(videos)
n = len(videos)
train_vids = videos[: int(0.8 * n)]
val_vids   = videos[int(0.8 * n): int(0.9 * n)]
test_vids  = videos[int(0.9 * n):]

train_df = df[df.video.isin(train_vids)].copy()
val_df   = df[df.video.isin(val_vids)].copy()
test_df  = df[df.video.isin(test_vids)].copy()

train_medians = train_df[feat_cols].replace([np.inf, -np.inf], np.nan).median(axis=0)

def sanitize(block_df):
    # replace cals we don't want with medians calculated only from training set
    X = block_df[feat_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(train_medians)
    block_df[feat_cols] = X
    return block_df

train_df = sanitize(train_df)
val_df   = sanitize(val_df)
test_df  = sanitize(test_df)

scaler = StandardScaler().fit(train_df[feat_cols].to_numpy(np.float32)) # only fitting on the training data!!

def scale_block(block_df):
    X = block_df[feat_cols].to_numpy(np.float32)
    X = scaler.transform(X).astype(np.float32)
    return X

def build_seq6(block_df, feat_cols, word2id, T=75):
    # grouping frames by video then scaling + fixing length + enforcing 6 word constant
    Xs, Ys, vids = [], [], []
    skipped = 0
    for vid, g in block_df.groupby("video"):
        g = g.sort_values("frame")
        X_raw = scale_block(g)
        X_fix = fix_length(X_raw, T)
        w6 = collapsed_words(g)
        if len(w6) != 6:
            skipped += 1
            continue
        y = np.array([word2id[w] for w in w6], dtype=np.int32)
        Xs.append(X_fix); Ys.append(y); vids.append(vid)
    print(f"[build_seq6] Skipped {skipped} clips that r not exactly 6 words.")
    if not Xs:
        raise ValueError("No clips left that r 6 words.")
    return np.stack(Xs, 0), np.stack(Ys, 0), np.array(vids)

def build_frames(block_df, feat_cols, word2id, T=75):
    # not using this builder but maybe for future development could be helpful
    # CHAT SUGGESTED! BUT WE DON'T USE THIS
    Xs, Ys = [], []
    for vid, g in block_df.groupby("video"):
        g = g.sort_values("frame")
        X_raw = scale_block(g)
        X_fix = fix_length(X_raw, T)
        y_raw = g["word"].astype(str).map(word2id).to_numpy(np.int64)
        if len(y_raw) > T:
            idx = np.linspace(0, len(y_raw)-1, T).round().astype(int)
            y = y_raw[idx]
        elif len(y_raw) < T:
            y = np.concatenate([y_raw, np.repeat(y_raw[-1], T - len(y_raw))])
        else:
            y = y_raw
        Xs.append(X_fix); Ys.append(y.astype(np.int32))
    return np.stack(Xs, 0), np.stack(Ys, 0)

X_train6, Y_train6, vids_train = build_seq6(train_df, feat_cols, word2id, T=T)
X_val6,   Y_val6,   vids_val   = build_seq6(val_df,   feat_cols, word2id, T=T)
X_test6,  Y_test6,  vids_test  = build_seq6(test_df,  feat_cols, word2id, T=T)

def _finite(*arrs):
    return all(np.isfinite(a).all() for a in arrs)
assert _finite(X_train6, Y_train6, X_val6, Y_val6, X_test6, Y_test6)

# --- start of the model w attention layer
# our model struct is BiGRU + Attention. The GRU layers give us contextual encoding we studied these links: 
inp = tf.keras.Input(shape=(T, len(feat_cols)))
x = layers.Bidirectional(layers.GRU(H1, return_sequences=True))(inp)
x = layers.Dropout(0.3)(x)
x = layers.Bidirectional(layers.GRU(H2, return_sequences=True))(x)
H = x #hidden states

# using attention mechanism for tensor alignment (custom Keras layers used for Luong-style attention)
# read this + chat https://machinelearningmastery.com/adding-a-custom-attention-layer-to-recurrent-neural-network-in-keras/
# !!!! USED CHATGPT to debug the plentiful amount of tensor alignment proiblems caused by Permute, Dot layers
att_logits = layers.TimeDistributed(layers.Dense(SLOTS))(H)
att = layers.Permute((2, 1))(att_logits)
att = layers.Softmax(axis=-1)(att)
H_slots = layers.Dot(axes=(2, 1))([att, H])

# we get a projection of the final slot features to the words
out = layers.TimeDistributed(layers.Dense(num_classes, activation="softmax"))(H_slots)
model = models.Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
)
model.summary()

# making class weights only for TRAIN labels (LOSS BALANCING)
flat_train = Y_train6.reshape(-1)
classes_present = np.unique(flat_train)

from sklearn.utils.class_weight import compute_class_weight
cw_present = compute_class_weight(
    class_weight="balanced",
    classes=classes_present,
    y=flat_train
)

class_weight = {int(c): float(w) for c, w in zip(classes_present, cw_present)}

# !!! SUGGESTED BY CHATGPT> Optional: inspect what’s missing (just info)
missing = sorted(set(range(num_classes)) - set(classes_present))
print(f"{len(missing)} classes missing from train (ok):",
      [id2word[m] for m in missing][:10], "..." if len(missing) > 10 else "")

# converting the class_weight to per-token sample_weight matrices > got help from chat here because of shape alignment issues + unfamiliarity with keras
def make_sample_weights(Y, class_weight_dict):
    # so Keras constraint > we had to create a (Batch,SLOTS) matrix for temporal weighting
    # I got a lot of logic from Stack Overflow/Keras documentation on sequence labeling weights
    W = np.ones_like(Y, dtype=np.float32)         # shape (B, 6)
    for c, w in class_weight_dict.items():
        W[Y == c] = w
    return W

sw_train = make_sample_weights(Y_train6, class_weight)
sw_val   = make_sample_weights(Y_val6,   class_weight)

# trained WITHOUT class_weight and use sample_weight instead 
cbs = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
]

history = model.fit(
    X_train6, Y_train6,
    sample_weight=sw_train,                               # <-- per-token weights (B,6)
    validation_data=(X_val6, Y_val6, sw_val),            # <-- include val weights too
    epochs=20, batch_size=32,
    callbacks=cbs,
    verbose=1
)


model.evaluate(X_test6, Y_test6, verbose=1)
