"""PREDICT with the models saved by train.py. Never trains.

    python predict.py --input "D:\\new_images"                          # folder of images
    python predict.py --input photo.jpg --burst 1                       # single image
    python predict.py --input "D:\\new_images" --labels "D:\\labels.csv"   # + compare with ACTUAL AQI
    python predict.py --input ... --model GNN+NODE                      # pick a specific pipeline
    python predict.py --input ... --city Biratnagar                     # set city if not in the path

Images are grouped into frames exactly like in training. With --labels (same CSV format as
training) the output shows actual vs predicted AQI, the error, and summary metrics.
Output: predictions.csv (+ actual_vs_predicted_predict.png when actual AQI is known).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from aqi_common import (aqi_band, attach_labels, build_enc, build_reg, build_sequences, dev,
                        extract, load_resnet, make_frames, metrics, plot_actual_vs_pred,
                        predict_tensor, scan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--labels", default=None, help="CSV with actual AQI, to compare against")
    ap.add_argument("--bundle", default="models/aqi_bundle.pt")
    ap.add_argument("--model", default=None, help="e.g. ResNet+KAN (default: winner from training)")
    ap.add_argument("--burst", type=int, default=None, help="images per frame (default: as trained)")
    ap.add_argument("--city", default=None, help="override the city/location")
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_known_args()[0]

    if not Path(a.bundle).exists():
        raise SystemExit(f"{a.bundle} not found. Run train.py first.")
    B = torch.load(a.bundle, map_location="cpu", weights_only=False)
    cfg = B["config"]
    target = cfg["target"]
    name = a.model or B["best"]
    if name not in B["models"]:
        raise SystemExit(f"unknown model {name!r}; available: {list(B['models'])}")
    M = B["models"][name]
    print(f"model: {name}  (target={target}, trained on {cfg['n_frames']} frames, split={cfg['split']})")

    df = scan(a.input, a.burst or cfg["burst"], require_cat=False)
    if target == "aqi":
        df["y"] = np.nan                       # ignore category words in paths; use the CSV only
        if a.labels:
            df = attach_labels(df, a.labels)
    if a.city:
        df["city"] = a.city
    net = load_resnet(B["resnet"])             # identical backbone to training
    feats, hz, ok = extract(df["path"].to_numpy(), cfg["size"], cfg["use_mask"], net)
    df = df[ok].reset_index(drop=True)
    X, H, Y, C, ts, G, E = make_frames(df, feats, hz, cfg["views"])
    H = np.where(np.isnan(H), B["haze_median"], H).astype(np.float32)
    S = build_sequences(X, C, ts, cfg["seq"], verbose=M["extractor"] == "SNN")
    if M["extractor"] == "SNN" and ts.isna().any():
        print("! no timestamps in filenames: SNN falls back to views-as-time (weaker)")

    d_in = X.shape[-1]
    fs = B["fs"]
    Xn = fs.transform(X.reshape(-1, d_in)).reshape(X.shape).astype(np.float32)
    Sn = fs.transform(S.reshape(-1, d_in)).reshape(S.shape).astype(np.float32)
    Hn = B["hs"].transform(H).astype(np.float32)
    Xi = Sn if M["extractor"] == "SNN" else Xn

    enc = build_enc(M["extractor"], M["d_in"], cfg["emb"]).to(dev)
    enc.load_state_dict(M["enc"]); enc.eval()
    reg = build_reg(M["regressor"], M["d_reg"]).to(dev)
    reg.load_state_dict(M["reg"]); reg.eval()
    with torch.no_grad():
        Z = F.relu(enc.enc(torch.as_tensor(Xi, device=dev))).cpu().numpy()
    Fz = M["zs"].transform(np.concatenate([Z, Hn], 1)).astype(np.float32)
    pred = predict_tensor(reg, Fz) * cfg["y_sd"] + cfg["y_mu"]

    names = cfg["categories"]
    res = pd.DataFrame(dict(frame=G, city=C, timestamp=ts.to_numpy(),
                            n_images=df.groupby("group", sort=False).size().to_numpy()))
    known = ~np.isnan(Y)
    if target == "aqi":
        pred = np.clip(pred, 0, None)
        res["predicted_aqi"] = np.round(pred, 1)
        res["predicted_band"] = [names[i] for i in aqi_band(pred)]
        if known.any():
            res["actual_aqi"] = np.where(known, Y, np.nan)
            res["error"] = np.round(pred - Y, 1)
            res["abs_error"] = res["error"].abs()
            res["actual_band"] = [names[i] if k else "" for i, k in zip(aqi_band(np.nan_to_num(Y)), known)]
            m = metrics(Y[known], pred[known], "aqi")
            print(f"\nACTUAL vs PREDICTED on {known.sum()} frames:  MAE={m['MAE']:.1f}  "
                  f"RMSE={m['RMSE']:.1f}  R2={m['R2']:.3f}  band accuracy={m['Acc']:.3f}")
            if known.sum() > 1:
                try:
                    plot_actual_vs_pred(Y[known], pred[known], "actual_vs_predicted_predict.png", name)
                    print("saved actual_vs_predicted_predict.png")
                except Exception as e:
                    print("plot skipped:", e)
    else:
        res["score"] = np.round(pred, 3)
        res["predicted_category"] = [names[int(np.clip(round(float(s)), 0, len(names) - 1))] for s in pred]
        if known.any():
            res["true_category"] = [names[int(y)] if k else "" for y, k in zip(np.nan_to_num(Y), known)]
            print(f"accuracy: {np.mean(res.predicted_category[known] == res.true_category[known]):.3f}")
    res.to_csv(a.out, index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 30, "display.max_columns", 20,
                           "display.max_colwidth", 38):
        print(res)
    print("saved", a.out)


if __name__ == "__main__":
    main()