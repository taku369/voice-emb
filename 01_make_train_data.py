"""
男性:低/中/高、女性:低/中/高の6クラス分類用のデータ
"""

import argparse
import glob
import pickle

import librosa
import numpy as np

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("jvs_dir", help="path/to/jvs_ver1_dir")
    parser.add_argument("--out_dir", default="./", help="path/to/out_dir")
    return parser.parse_args()

def make_jvs_feature(jvs_dir, speaker_id):
    """
    speakerの全wav fileをMFCCで特徴ベクトルに

    Args:
        jvs_dir: string
            path to jvs_dir
        speaker_id: string
            jvs speaker id (ex. jvs001)
    
    Return:
        X: ndarray of shape(voice_num, 20)
            features of voice
    """

    wav_files = glob.glob(f"{jvs_dir}/{speaker_id}/*/wav24kHz16bit/*.wav")

    X = []
    for wav_file in wav_files:
        x, fs = librosa.load(wav_file, sr=24000)
        mfccs = librosa.feature.mfcc(x, sr=fs)  # (n_mfcc, sr*duration/hop_length)
        mfccs = np.mean(mfccs, axis=1)
        X.append(mfccs)

    return np.array(X)

args = parse()

X = []
y = []
# 手動で選んだ男低/男中/男高/女低/女中/女高
for label, speaker_id in enumerate(["jvs006", "jvs074", "jvs098", "jvs091", "jvs096", "jvs010"]):
    print(f"{label}:{speaker_id}...")
    tmp_X = make_jvs_feature(args.jvs_dir, speaker_id)
    tmp_y = np.array([label] * len(tmp_X))
    X.append(tmp_X)
    y.append(tmp_y)

X = np.concatenate(X)
y = np.concatenate(y)

# 保存
with open(f"{args.out_dir}/X.pkl", "wb") as fout:
    pickle.dump(X, fout)
with open(f"{args.out_dir}/y.pkl", "wb") as fout:
    pickle.dump(y, fout)