import argparse

import numpy as np
import pandas as pd

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="path/to/gender_f0range.txt")
    return parser.parse_args()

args = parse()

df = pd.read_csv(args.file_path, sep=" ")

m_df = df[df["Male_or_Female"] == "M"]
f_df = df[df["Male_or_Female"] == "F"]

m_inds = np.argsort(m_df["maxf0[Hz]"])
print("Male low Top3")
print(m_df.iloc[m_inds[:3]])
print("Male high Top3")
print(m_df.iloc[m_inds[::-1][:3]])
print("Male middle3")
middle = len(m_inds) // 2
print(m_df.iloc[m_inds[middle - 1: middle + 2]])
print("-----")

f_inds = np.argsort(f_df["maxf0[Hz]"])
print("Female low Top3")
print(f_df.iloc[f_inds[:3]])
print("Female high Top3")
print(f_df.iloc[f_inds[::-1][:3]])
print("Female middle3")
middle = len(f_inds) // 2
print(f_df.iloc[f_inds[middle - 1: middle + 2]])