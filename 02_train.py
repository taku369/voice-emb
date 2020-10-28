import argparse
import pickle

from sklearn.neural_network import MLPClassifier

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./", help="path/to/out_dir")
    parser.add_argument("--X_path", default="X.pkl", help="path/to/X.pkl")
    parser.add_argument("--y_path", default="y.pkl", help="path/to/y.pkl")
    return parser.parse_args()


args = parse()

with open(args.X_path, "rb") as fin:
    X = pickle.load(fin)
with open(args.y_path, "rb") as fin:
    y = pickle.load(fin)

model = MLPClassifier(hidden_layer_sizes=(20, 30), verbose=True, max_iter=1000)
model.fit(X, y)

with open(f"{args.out_dir}/model.pkl", "wb") as fout:
    pickle.dump(model, fout)