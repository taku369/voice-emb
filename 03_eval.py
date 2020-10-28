import argparse
import glob
import pickle

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSES = (
    "male low", 
    "male middle", 
    "male high", 
    "female low", 
    "feamale middle", 
    "female high"
)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("jvs_dir", help="path/to/jvs_ver1_dir")
    parser.add_argument("--out_dir", default="./", help="path/to/out_dir")
    parser.add_argument("--model_path", default="model.pkl", help="path/to/model.pkl")
    return parser.parse_args()


def get_feature(wav_file):
    """
    音声ファイルの特徴ベクトルを抽出

    Args:
        wav_file: string
            path to wave file

    Returns:
        mfccs: ndarray of shape(20,)
            feature of wave file
    """
    x, fs = librosa.load(wav_file, sr=24000)
    mfccs = librosa.feature.mfcc(x, sr=fs)  # (n_mfcc, sr*duration/hop_length)
    mfccs = np.mean(mfccs, axis=1)
    return mfccs


def get_proba(df, jvs_dir, model, trained_speaker):
    speakers = df.sort_values("maxf0[Hz]")["speaker"]  # maxf0低い順

    # 各話者ごとに音声ファイルを一つ選んで特徴ベクトルを抽出
    X = []
    for speaker in speakers:
        if speaker not in trained_speaker:  # 学習に使った話者は除く
            wav_file = glob.glob(f"{jvs_dir}/{speaker}/nonpara30/wav24kHz16bit/*.wav")[0]
            X.append(get_feature(wav_file))

    # 推論
    X = np.array(X)
    proba = model.predict_proba(X)  # ndarray of shape (n_samples:話者数, n_classes:6) 

    return proba


def draw_heatmap(arr, title, out_dir, xlabel=CLASSES):
    """
    https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    row, col = arr.shape
    fig, ax = plt.subplots(figsize=(col, row))
    im = ax.imshow(arr)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xlabel)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xlabel)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for r in range(row):
        for c in range(col):
            text = ax.text(c, r, f"{arr[r, c]:.2f}", ha="center", va="center", color="w")

    ax.set_title(title)

    fig.savefig(f"{out_dir}/{title}.png", bbox_inches='tight')


args = parse()

with open(args.model_path, "rb") as fin:
    model = pickle.load(fin)

df = pd.read_csv(f"{args.jvs_dir}/gender_f0range.txt", sep=" ")

# 男性
print("---------Male----------")
m_df = df[df["Male_or_Female"] == "M"]
m_proba = get_proba(m_df, args.jvs_dir, model, trained_speaker=("jvs006", "jvs074", "jvs098"))

# for prob in m_proba:
#     print(" ".join([f"{p:.2f}" for p in prob]))

draw_heatmap(m_proba, "male_proba", args.out_dir)

# 女性
print("---------Female----------")
f_df = df[df["Male_or_Female"] == "F"]
f_proba = get_proba(f_df, args.jvs_dir, model, trained_speaker=("jvs091", "jvs096", "jvs010"))

# for prob in f_proba:
#     print(" ".join([f"{p:.2f}" for p in prob]))

draw_heatmap(f_proba, "female_proba", args.out_dir)