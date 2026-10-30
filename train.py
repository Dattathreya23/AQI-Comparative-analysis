"""TRAIN once. Trains the 6 pipelines, evaluates them, saves everything needed for
inference to models/aqi_bundle.pt. Afterwards use predict.py (it never trains).

  With a label table (real AQI):      python train.py --root "D:\\dataset" --labels "D:\\labels.csv"
  Without (category from path words): python train.py --root "D:\\dataset"
Options: --split random|time|city  --holdout_city NAME  --no_mask  --seeds 5  --recompute

Label table columns used: Filename, AQI (required); Location, Year, Month, Day, Hour (recommended).
PM2.5/PM10/O3/CO/SO2/NO2 are deliberately NOT used as inputs: AQI is computed from them, so
feeding them in would leak the answer. Only images are used to predict.
Only the first seed's models are saved; --seeds > 1 just adds mean +- std to the report.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from aqi_common import (AQI_BANDS, CAT_NAMES, EXTRACTORS, HAZE_NAMES, REGRESSORS, aqi_band,
                        attach_labels, build_enc, build_reg, build_sequences, dev, extract, fit,
                        load_resnet, make_frames, metrics, plot_actual_vs_pred, predict_tensor, scan)

DEFAULT_ROOT = r"D:\Desktop\MP\dataset"          # <- change to your image folder
DEFAULT_LABELS = None                            # <- or r"D:\Desktop\MP\labels.csv"


def get_features(df, a, out):
    """Returns features + the exact ResNet weights that produced them (saved in the bundle)."""
    key = out / f"cache_s{a.size}_m{int(not a.no_mask)}.npz"
    bb = key.with_suffix(".backbone.pt")
    paths = df["path"].to_numpy()
    if key.exists() and bb.exists() and not a.recompute:
        z = np.load(key, allow_pickle=True)
        if np.array_equal(z["paths"], paths):
            print("loaded cached features:", key)
            return z["feats"], z["hz"], z["ok"], torch.load(bb)
    net = load_resnet()
    feats, hz, ok = extract(paths, a.size, not a.no_mask, net)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    np.savez(key, feats=feats, hz=hz, ok=ok, paths=paths)
    torch.save(state, bb)
    return feats, hz, ok, state


def split_indices(a, C, ts, E, seed):
    """Split by EVENT (place + time = one AQI label), never by image or by frame, so frames
    sharing a label can't land on both sides of the train/test boundary."""
    rng = np.random.RandomState(seed)
    ue, inv = np.unique(E, return_inverse=True)
    ev = pd.DataFrame({"e": inv, "c": C, "t": ts.values}).groupby("e").agg(c=("c", "first"), t=("t", "min"))
    ne = len(ev)

    def frames_of(ev_ids):
        return np.where(np.isin(inv, ev_ids))[0]

    if a.holdout_city or a.split == "city":
        if not a.holdout_city:
            raise SystemExit("--split city needs --holdout_city NAME")
        te_ev = ev.index[ev.c == a.holdout_city].to_numpy()
        if len(te_ev) == 0:
            raise SystemExit(f"city {a.holdout_city!r} not found; cities: {sorted(set(C))}")
        rest = rng.permutation(ev.index[ev.c != a.holdout_city].to_numpy())
        nv = max(1, int(0.15 * len(rest)))
        return frames_of(rest[nv:]), frames_of(rest[:nv]), frames_of(te_ev)
    if a.split == "time":
        if ev.t.isna().any():
            raise SystemExit("--split time needs a timestamp in every filename")
        tr, va, te = [], [], []
        for c, g in ev.groupby("c"):                # chronological within each city
            ids = g.sort_values("t").index.to_numpy()
            k1, k2 = int(0.70 * len(ids)), int(0.85 * len(ids))
            tr += list(ids[:k1]); va += list(ids[k1:k2]); te += list(ids[k2:])
        return frames_of(tr), frames_of(va), frames_of(te)
    perm = rng.permutation(ne)
    nt, nv = max(1, int(0.2 * ne)), max(1, int(0.15 * ne))
    return frames_of(perm[nt + nv:]), frames_of(perm[nt:nt + nv]), frames_of(perm[:nt])


def cpu_state(m):
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}


def train_all(a, seed, target, Xn, Sn, Hn, Y, mu, sd, tr, va, te):
    torch.manual_seed(seed); np.random.seed(seed)
    Yn = ((Y - mu) / sd).astype(np.float32)          # models learn standardised targets
    rows, arts, preds = [], {}, {}
    inputs = {"ResNet": Xn, "GNN": Xn, "SNN": Sn}
    for en in EXTRACTORS:
        Xi = inputs[en]
        net = fit(build_enc(en, Xi.shape[-1], a.emb),
                  (Xi[tr], Yn[tr]), (Xi[va], Yn[va]), epochs=80, lr=1e-3)
        with torch.no_grad():
            Z = F.relu(net.enc(torch.as_tensor(Xi, device=dev))).cpu().numpy()
        Fz0 = np.concatenate([Z, Hn], 1)
        zs = StandardScaler().fit(Fz0[tr])
        Fz = zs.transform(Fz0).astype(np.float32)
        names = [f"{en}_emb{i}" for i in range(a.emb)] + HAZE_NAMES
        for rn in REGRESSORS:
            reg = fit(build_reg(rn, Fz.shape[1]), (Fz[tr], Yn[tr]), (Fz[va], Yn[va]),
                      epochs=300, lr=3e-3, bs=128)
            name = f"{en}+{rn}"
            p = predict_tensor(reg, Fz[te]) * sd + mu
            if target == "aqi":
                p = np.clip(p, 0, None)
            m = metrics(Y[te], p, target)
            rows.append(dict(seed=seed, model=name, **m))
            preds[name] = p
            arts[name] = dict(extractor=en, regressor=rn, d_in=Xi.shape[-1], d_reg=Fz.shape[1],
                              enc=cpu_state(net), reg=cpu_state(reg), zs=zs, names=names,
                              _reg=reg, _Fz=Fz)                    # _* kept for SHAP, not saved
            print(f"  seed {seed} {name:12s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
    return rows, arts, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--labels", default=DEFAULT_LABELS, help="CSV with real AQI (Filename, AQI, ...)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--burst", type=int, default=30, help="max images per frame")
    ap.add_argument("--views", type=int, default=8, help="views sampled per frame (ResNet/GNN)")
    ap.add_argument("--seq", type=int, default=4, help="frames per SNN time sequence")
    ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--split", choices=["random", "time", "city"], default="random")
    ap.add_argument("--holdout_city", default=None)
    ap.add_argument("--no_mask", action="store_true")
    ap.add_argument("--recompute", action="store_true", help="ignore cached features")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models_dir", default="models")
    a = ap.parse_known_args()[0]
    out, mdir = Path(a.out), Path(a.models_dir)
    out.mkdir(exist_ok=True); mdir.mkdir(exist_ok=True)
    target = "aqi" if a.labels else "category"

    df = scan(a.root, a.burst, require_cat=not a.labels)
    if a.labels:
        df = attach_labels(df, a.labels)
    print(f"{len(df)} images | cities {sorted(df.city.unique())} | target={target} | "
          f"timestamps parsed for {df.ts.notna().mean():.0%}")
    if target == "aqi":
        print(f"AQI range {df.y.min():.0f}-{df.y.max():.0f}, mean {df.y.mean():.0f}; images per band "
              f"{ {AQI_BANDS[int(k)]: int(v) for k, v in zip(*np.unique(aqi_band(df.y), return_counts=True))} }")
    else:
        print("categories", dict(df.y.value_counts().sort_index()))
    feats, hz, ok, backbone = get_features(df, a, out)
    df = df[ok].reset_index(drop=True)
    X, H, Y, C, ts, G, E = make_frames(df, feats, hz, a.views)
    print(f"{len(Y)} frames in {len(np.unique(E))} events (place+time)")
    if np.unique(Y).size < 2:
        raise SystemExit("Only one label value present: nothing to learn.")
    if len(np.unique(E)) < 20:
        print("! very few events; scores will be noisy")
    S = build_sequences(X, C, ts, a.seq)

    all_rows, first = [], None
    for k in range(a.seeds):
        seed = a.seed + k
        tr, va, te = split_indices(a, C, ts, E, seed)
        if min(len(tr), len(va), len(te)) == 0:
            raise SystemExit(f"empty split (train/val/test = {len(tr)}/{len(va)}/{len(te)}): need more data")
        mu, sd = (float(Y[tr].mean()), float(Y[tr].std() or 1.0)) if target == "aqi" else (0.0, 1.0)
        if k == 0:
            print(f"frames train/val/test = {len(tr)}/{len(va)}/{len(te)} | split={a.split}")
            if target == "aqi":
                print(f"test AQI range {Y[te].min():.0f}-{Y[te].max():.0f} (train {Y[tr].min():.0f}-{Y[tr].max():.0f})")
        fs = StandardScaler().fit(X[tr].reshape(-1, X.shape[-1]))
        Xn = fs.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
        Sn = fs.transform(S.reshape(-1, S.shape[-1])).reshape(S.shape).astype(np.float32)
        hs = StandardScaler().fit(H[tr])
        Hn = hs.transform(H).astype(np.float32)
        rows, arts, preds = train_all(a, seed, target, Xn, Sn, Hn, Y, mu, sd, tr, va, te)
        all_rows += rows
        if k == 0:
            first = dict(arts=arts, preds=preds, fs=fs, hs=hs, mu=mu, sd=sd,
                         haze_median=np.median(H[tr], 0), tr=tr, te=te)

    res = pd.DataFrame(all_rows)
    res.to_csv(out / "results_all_seeds.csv", index=False)
    summ = res.groupby("model")[["R2", "RMSE", "MAE", "Acc"]].agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ = summ.sort_values(["R2_mean", "RMSE_mean"], ascending=[False, True])
    summ.to_csv(out / "results_summary.csv")
    unit = "AQI points" if target == "aqi" else "category-score units"
    print(f"\n=== 6-model benchmark on test set (RMSE/MAE in {unit}; mean +- std over seeds) ===")
    print(summ.round(3).to_string())
    best = summ.index[0]
    print(f"\nWinner: {best}")

    # ---- actual vs predicted on the test set (first seed) ----
    te = first["te"]
    cmp = pd.DataFrame({"frame": G[te], "city": C[te], "timestamp": ts.iloc[te].to_numpy(), "actual": Y[te]})
    for n, p in first["preds"].items():
        cmp[f"pred_{n}"] = np.round(p, 2)
    cmp["error_best"] = np.round(first["preds"][best] - Y[te], 2)
    cmp.sort_values(["city", "timestamp"]).to_csv(out / "test_predictions.csv", index=False)
    print("saved", out / "test_predictions.csv", "(actual vs predicted for every test frame)")
    if target == "aqi":
        try:
            plot_actual_vs_pred(Y[te], first["preds"][best], out / "actual_vs_predicted.png", f"{best} (test set)")
            print("saved", out / "actual_vs_predicted.png")
        except Exception as e:
            print("plot skipped:", e)

    # ---- save everything inference needs (no training happens after this point) ----
    saved = {n: {k: v for k, v in d.items() if not k.startswith("_")} for n, d in first["arts"].items()}
    bundle = dict(
        config=dict(size=a.size, use_mask=not a.no_mask, views=a.views, seq=a.seq, emb=a.emb,
                    burst=a.burst, split=a.split, holdout_city=a.holdout_city, target=target,
                    y_mu=first["mu"], y_sd=first["sd"],
                    trained_on=str(a.root), n_frames=int(len(Y)),
                    categories=AQI_BANDS if target == "aqi" else CAT_NAMES),
        resnet=backbone, fs=first["fs"], hs=first["hs"], haze_median=first["haze_median"],
        models=saved, best=best, summary=summ.reset_index().to_dict("records"))
    torch.save(bundle, mdir / "aqi_bundle.pt")
    print(f"saved trained models -> {mdir / 'aqi_bundle.pt'}  (use predict.py from now on)")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        d = first["arts"][best]
        reg, Fz = d["_reg"], d["_Fz"]
        f = lambda z: predict_tensor(reg, z.astype(np.float32))
        ex = shap.KernelExplainer(f, shap.kmeans(Fz[first["tr"]], min(20, len(first["tr"]))))
        sub = Fz[first["te"]][:100]
        sv = ex.shap_values(sub, nsamples=200)
        shap.summary_plot(sv, sub, feature_names=d["names"], show=False)
        plt.tight_layout(); plt.savefig(out / "shap_summary.png", dpi=200); plt.close()
        print("saved", out / "shap_summary.png")
    except Exception as e:
        print("SHAP skipped:", e)


if __name__ == "__main__":
    main()