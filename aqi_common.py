"""Shared code for train.py and predict.py: parsing, sky mask, feature extraction,
frame building, model definitions. Nothing in here trains by itself except fit()."""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CATS = {"good": 0, "satisfactory": 1, "moderate": 2, "poor": 3, "verypoor": 4, "severe": 5}
CAT_NAMES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CAT_RE = re.compile(r"(?<![a-z])(very[\s_\-]*poor|severe|poor|moderate|satisfactory|good)(?![a-z])", re.I)
TS_RE = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})[-_ T]+(\d{2})[.:h_\-]?(\d{2})")
AQI_BANDS = ["Good", "Moderate", "Unhealthy for Sensitive", "Unhealthy", "Very Unhealthy", "Hazardous"]


def aqi_band(x):
    """US-EPA AQI band index 0..5 (0-50, 51-100, 101-150, 151-200, 201-300, 301+)."""
    return np.digitize(np.asarray(x, dtype=float), [50, 100, 150, 200, 300], right=True)
HAZE_NAMES = ["sky_dark_channel", "sky_contrast", "sky_blue_ratio", "sky_brightness", "sky_saturation"]
dev = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================ data / parsing ====
def parse_path(rel, require_cat=True):
    """rel = path relative to root. Category is optional when require_cat=False."""
    names = list(rel.parts[:-1]) + [rel.stem]
    cat = None
    for n in reversed(names):                      # filename first, then folders upward
        m = CAT_RE.search(n)
        if m:
            cat = CATS[re.sub(r"[\s_\-]", "", m.group(1)).lower()]
            break
    if cat is None and require_cat:
        return None
    stem = rel.stem
    first = CAT_RE.sub("", rel.parts[0]).strip(" _-") if len(rel.parts) > 1 else ""
    city = first or stem.split("_")[0]
    ts = pd.NaT
    for n in [stem] + list(reversed(rel.parts[:-1])):
        m = TS_RE.search(n)
        if m:
            try:
                ts = pd.Timestamp(*[int(g) for g in m.groups()])
                break
            except ValueError:
                pass
    mi = re.search(r"[-_](\d+)$", stem)
    return dict(city=city, y=np.nan if cat is None else cat, ts=ts,
                frame=str(rel.parent / re.sub(r"[-_]\d+$", "", stem)),
                idx=int(mi.group(1)) if mi else 0)


def scan(root, burst, require_cat=True, city=None):
    root = Path(root)
    if not root.exists():
        raise SystemExit(f"path does not exist: {root}")
    if root.is_file():
        files, base = [root], root.parent
    else:
        files, base = list(root.rglob("*")), root
    recs, n_img = [], 0
    for p in files:
        if p.suffix.lower() in EXTS:
            n_img += 1
            r = parse_path(p.relative_to(base), require_cat)
            if r:
                r["path"] = str(p)
                if city:
                    r["city"] = city
                recs.append(r)
    if not recs:
        ex = [str(p.relative_to(base)) for p in files[:5]]
        raise SystemExit(f"No usable images under {root}\n"
                         f"image files found: {n_img}; first entries: {ex}\n"
                         f"(training needs a category word like Good/Moderate/Poor in the path)")
    df = pd.DataFrame(recs).sort_values(["frame", "idx", "path"]).reset_index(drop=True)
    df["group"] = df["frame"] + "#" + (df.groupby("frame").cumcount() // burst).astype(str)
    return df


def _norm(s):
    return re.sub(r"\s+", "", str(s)).lower()


def _row_ts(r):
    try:
        hh, _, mm = str(r["Hour"]).strip().partition(".")
        mm = (mm + "0")[:2] if mm else "00"
        return pd.Timestamp(int(r["Year"]), int(r["Month"]), int(r["Day"]), int(hh or 0), int(mm))
    except Exception:
        return pd.NaT


def attach_labels(df, csv_path):
    """Join real AQI (+ location) from the label table onto the image table.
    1) exact match on filename (whitespace/case ignored)
    2) fallback: same timestamp, if that timestamp has a single (location, AQI) in the table."""
    L = pd.read_csv(csv_path, dtype={"Hour": str}, encoding="utf-8-sig")
    L.columns = [c.strip() for c in L.columns]
    for need in ("Filename", "AQI"):
        if need not in L.columns:
            raise SystemExit(f"label file needs a '{need}' column; has {list(L.columns)}")
    L["key"] = L["Filename"].map(_norm)
    has_ts = all(c in L.columns for c in ("Year", "Month", "Day", "Hour"))
    L["_ts"] = L.apply(_row_ts, axis=1) if has_ts else pd.NaT
    L["_loc"] = L["Location"].astype(str).str.strip() if "Location" in L.columns else "unknown"
    exact = L.drop_duplicates("key").set_index("key")
    out = df.copy()
    keys = out["path"].map(lambda p: _norm(Path(p).name))
    out["y"] = keys.map(exact["AQI"]).astype(float)
    out["_loc"] = keys.map(exact["_loc"]).astype(object)
    n_exact = int(out["y"].notna().sum())
    if has_ts:
        Lt = L.dropna(subset=["_ts"])
        # pass A: filename prefix (e.g. BRI / BIR) -> location, learned from the exact matches;
        #         then match on (location, timestamp)
        pref = out["path"].map(lambda p: Path(p).stem.split("_")[0].lower())
        hit = out["y"].notna()
        pm = {}
        for p, ls in out.loc[hit, "_loc"].groupby(pref[hit]):
            if ls.nunique() == 1:
                pm[p] = ls.iloc[0]
        g2 = (Lt.groupby(["_loc", "_ts"]).agg(n=("AQI", "nunique"), aqi=("AQI", "first"))
              .query("n == 1").reset_index())
        miss = out["y"].isna() & out["ts"].notna()
        if miss.any() and pm:
            m = pd.DataFrame({"ts": out.loc[miss, "ts"].to_numpy(),
                              "_loc": pref[miss].map(pm).astype(object).to_numpy()})
            m = m.merge(g2, left_on=["_loc", "ts"], right_on=["_loc", "_ts"], how="left")
            ok = m["aqi"].notna().to_numpy()
            idx = out.index[miss][ok]
            out.loc[idx, "y"] = m.loc[ok, "aqi"].astype(float).to_numpy()
            out.loc[idx, "_loc"] = m.loc[ok, "_loc"].to_numpy()
        # pass B: timestamp alone, only if that timestamp has a single location/AQI in the table
        g = (Lt.groupby("_ts").agg(n=("AQI", "nunique"), nl=("_loc", "nunique"),
                                   aqi=("AQI", "first"), loc=("_loc", "first")).query("n == 1 and nl == 1"))
        miss = out["y"].isna() & out["ts"].notna()
        if miss.any():
            out.loc[miss, "y"] = out.loc[miss, "ts"].map(g["aqi"]).astype(float)
            out.loc[miss, "_loc"] = out.loc[miss, "ts"].map(g["loc"]).astype(object)
    n_fb = int(out["y"].notna().sum()) - n_exact
    n_drop = int(out["y"].isna().sum())
    print(f"labels: {n_exact} images matched by filename, {n_fb} by timestamp, "
          f"{n_drop} unmatched (dropped)")
    if n_drop:
        print("  e.g. unmatched:", [Path(p).name for p in out.loc[out.y.isna(), "path"].head(3)])
    out = out[out["y"].notna()].copy()
    out["city"] = out["_loc"].fillna("unknown").astype(str)
    return out.drop(columns="_loc").reset_index(drop=True)


def plot_actual_vs_pred(actual, pred, path, title=""):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    lim = [0, max(actual.max(), pred.max()) * 1.05]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(actual, pred, s=22, alpha=0.75)
    ax.plot(lim, lim, "k--", lw=1, label="perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual AQI"); ax.set_ylabel("Predicted AQI")
    ax.set_title(f"{title}\nMAE={mean_absolute_error(actual, pred):.1f}  "
                 f"RMSE={np.sqrt(mean_squared_error(actual, pred)):.1f}  "
                 f"R2={r2_score(actual, pred):.3f}", fontsize=10)
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ====================================================== sky mask / haze stats ====
def sky_mask(a):
    """Sky = smooth, bright region connected to the top of the image. Works for blue AND
    grey/hazy skies (a blue-only rule would discard exactly the polluted images)."""
    gray = a.mean(-1)
    mu = ndi.uniform_filter(gray, 7)
    sd = np.sqrt(np.clip(ndi.uniform_filter(gray ** 2, 7) - mu ** 2, 0, None))
    cand = (sd < 0.025) & (gray > 0.35)
    cand[int(0.75 * a.shape[0]):] = False
    lab, n = ndi.label(cand)
    if n == 0:
        return cand
    keep = np.unique(lab[:3][lab[:3] > 0])
    return np.isin(lab, keep)


def haze_stats(a, m):
    if m.mean() < 0.02:
        return [np.nan] * len(HAZE_NAMES)
    dc = ndi.minimum_filter(a.min(-1), size=15)[m].mean()
    gray = a.mean(-1)[m]
    mx, mn = a.max(-1)[m], a.min(-1)[m]
    br = (a[..., 2][m] / (a.sum(-1)[m] + 1e-6)).mean()
    return [float(dc), float(gray.std()), float(br), float(gray.mean()),
            float(((mx - mn) / (mx + 1e-6)).mean())]


def load_resnet(state=None):
    """state = exact backbone weights saved at training time (guarantees identical features
    in predict.py, even if the pretrained download is unavailable)."""
    import torchvision
    if state is not None:
        net = torchvision.models.resnet18(weights=None)
        net.fc = nn.Identity()
        net.load_state_dict(state)
        return net.eval().to(dev)
    try:
        net = torchvision.models.resnet18(weights="DEFAULT")
    except Exception as e:
        print(f"! pretrained ResNet18 unavailable ({type(e).__name__}); using random init, "
              f"expect weaker results")
        net = torchvision.models.resnet18(weights=None)
    net.fc = nn.Identity()
    return net.eval().to(dev)


@torch.no_grad()
def extract(paths, size, use_mask, net, bs=32):
    """Stream images -> (resnet features, haze stats, ok-mask). Keeps RAM low."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    feats, hz, ok, batch = [], [], np.zeros(len(paths), bool), []

    def flush():
        x = torch.from_numpy(np.stack(batch)).to(dev).permute(0, 3, 1, 2).float()
        feats.append(net((x - mean) / std).cpu().numpy())
        batch.clear()

    for i, p in enumerate(paths):
        try:
            a = np.asarray(Image.open(p).convert("RGB").resize((size, size)), dtype=np.float32) / 255
        except Exception as e:
            print("skip unreadable:", p, e)
            continue
        m = sky_mask(a)
        hz.append(haze_stats(a, m))
        if use_mask and m.mean() >= 0.02:      # paint non-sky with the mean sky colour
            a = np.where(m[..., None], a, a[m].mean(0))
        batch.append(a); ok[i] = True
        if len(batch) == bs:
            flush()
        if (i + 1) % 500 == 0:
            print(f"  features: {i + 1}/{len(paths)}")
    if batch:
        flush()
    if not feats:
        raise SystemExit("no readable images")
    return np.concatenate(feats), np.array(hz, dtype=np.float32), ok


# ================================================================== frames ====
def make_frames(df, feats, hz, V):
    """One sample per frame: V views, mean haze stats, label, city, timestamp."""
    X, H, Y, C, T, G, E = [], [], [], [], [], [], []
    for g, d in df.groupby("group", sort=False):
        idx = d.index.to_numpy()
        pick = idx[np.linspace(0, len(idx) - 1, V).round().astype(int)]
        X.append(feats[pick])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H.append(np.nanmean(hz[idx], 0))
        t = d.ts.min()
        Y.append(d.y.iloc[0]); C.append(d.city.iloc[0]); T.append(t); G.append(g)
        # EVENT = same place + same time = same AQI label; splits must never cut through one
        E.append(f"{d.city.iloc[0]}|{t}" if pd.notna(t) else g)
    X, H = np.stack(X), np.stack(H)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(H, 0)
    H = np.where(np.isnan(H), med, H)
    return (X, H.astype(np.float32), np.array(Y, np.float32), np.array(C),
            pd.to_datetime(pd.Series(T)), np.array(G), np.array(E))


def build_sequences(X, city, ts, T, verbose=True):
    """SNN input. With timestamps: the T most recent frames of the same city (ending at
    this frame, padded with the earliest). Without: T views of the frame itself."""
    N, V, D = X.shape
    pooled = X.mean(1)
    S = np.zeros((N, T, D), np.float32)
    n_temporal = 0
    for c in np.unique(city):
        ids = np.where(city == c)[0]
        if ts.iloc[ids].notna().all() and len(ids) > 1:
            order = ids[np.argsort(ts.iloc[ids].to_numpy(), kind="stable")]
            for pos, i in enumerate(order):
                win = order[max(0, pos - T + 1):pos + 1]
                win = np.concatenate([np.repeat(win[:1], T - len(win)), win])
                S[i] = pooled[win]
            n_temporal += len(ids)
        else:
            for i in ids:
                S[i] = X[i][np.linspace(0, V - 1, T).round().astype(int)]
    if verbose:
        print(f"SNN sequences: {n_temporal}/{N} frames use real timestamp order, "
              f"{N - n_temporal} fall back to views-as-time")
    return S


# ============================================================== extractors ====
class VisualEnc(nn.Module):          # ResNet track: pooled per-view visual features
    def __init__(s, d, D):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, D))

    def forward(s, x):
        return s.f(x.mean(1))


class GraphEnc(nn.Module):           # GNN track: views = nodes, kNN cosine graph, 2 GCN layers
    def __init__(s, d, D, k=3):
        super().__init__()
        s.k = k
        s.p = nn.Linear(d, 128)
        s.w1, s.w2 = nn.Linear(128, 128), nn.Linear(128, 128)
        s.o = nn.Linear(128, D)

    def adj(s, h):
        hn = F.normalize(h, dim=-1)
        sim = hn @ hn.transpose(1, 2)
        k = min(s.k + 1, sim.shape[-1])
        A = torch.zeros_like(sim).scatter_(-1, sim.topk(k, dim=-1).indices, 1.0)
        A = ((A + A.transpose(1, 2)) > 0).float()
        return A / A.sum(-1, keepdim=True)

    def forward(s, x):
        h = F.relu(s.p(x))
        A = s.adj(h.detach())
        h = F.relu(s.w1(A @ h)) + h
        h = F.relu(s.w2(A @ h)) + h
        return s.o(h.mean(1))


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1 + 10 * x.abs()) ** 2


class SpikingEnc(nn.Module):         # SNN track: leaky integrate-and-fire over the time axis
    def __init__(s, d, D, beta=0.8, thr=1.0):
        super().__init__()
        s.p, s.rec, s.o = nn.Linear(d, 128), nn.Linear(128, 128), nn.Linear(256, D)
        s.beta, s.thr = beta, thr

    def forward(s, x):
        B, T, _ = x.shape
        mem = torch.zeros(B, 128, device=x.device)
        spk, count = torch.zeros_like(mem), torch.zeros_like(mem)
        for t in range(T):
            mem = s.beta * mem + s.p(x[:, t]) + s.rec(spk)
            spk = _Spike.apply(mem - s.thr)
            mem = mem - spk * s.thr
            count = count + spk
        return s.o(torch.cat([count / T, mem], -1))


class EncHead(nn.Module):
    def __init__(s, enc, D):
        super().__init__()
        s.enc, s.head = enc, nn.Linear(D, 1)

    def forward(s, x):
        return s.head(F.relu(s.enc(x))).squeeze(-1)


EXTRACTORS = {"ResNet": VisualEnc, "GNN": GraphEnc, "SNN": SpikingEnc}


# ============================================================== regressors ====
class KAN(nn.Module):
    """Compact KAN: every edge is a learnable 1-D function (RBF-basis spline + SiLU base)."""

    class Layer(nn.Module):
        def __init__(s, i, o, grid=8, lo=-3.0, hi=3.0):
            super().__init__()
            s.register_buffer("grid", torch.linspace(lo, hi, grid))
            s.h = (hi - lo) / (grid - 1)
            s.spline = nn.Linear(i * grid, o, bias=False)
            s.base = nn.Linear(i, o)

        def forward(s, x):
            phi = torch.exp(-(((x[..., None] - s.grid) / s.h) ** 2))
            return s.spline(phi.flatten(1)) + s.base(F.silu(x))

    def __init__(s, d, hidden=16):
        super().__init__()
        s.l1, s.n, s.l2 = KAN.Layer(d, hidden), nn.LayerNorm(hidden), KAN.Layer(hidden, 1)

    def forward(s, x):
        return s.l2(s.n(s.l1(x))).squeeze(-1)


class NODE(nn.Module):
    """Compact NODE: ensemble of differentiable oblivious decision trees."""

    def __init__(s, d, trees=64, depth=4):
        super().__init__()
        s.sel = nn.Parameter(torch.randn(trees, depth, d) * 0.1)
        s.thr = nn.Parameter(torch.randn(trees, depth) * 0.5)
        s.logt = nn.Parameter(torch.zeros(trees, depth))
        s.leaf = nn.Parameter(torch.randn(trees, 2 ** depth) * 0.1)
        s.bias = nn.Parameter(torch.zeros(1))
        s.register_buffer("codes", torch.tensor(
            [[(l >> k) & 1 for k in range(depth)] for l in range(2 ** depth)]).float())

    def forward(s, x):
        xs = torch.einsum("bf,tdf->btd", x, F.softmax(s.sel, -1))
        p = torch.sigmoid((xs - s.thr) * torch.exp(-s.logt))[:, :, None, :]
        leafp = (p * s.codes + (1 - p) * (1 - s.codes)).prod(-1)
        return (leafp * s.leaf).sum(-1).mean(-1) * 4 + s.bias


REGRESSORS = {"KAN": KAN, "NODE": NODE}


def build_enc(name, d_in, emb):
    return EncHead(EXTRACTORS[name](d_in, emb), emb)


def build_reg(name, d):
    return REGRESSORS[name](d)


# ======================================================== training / metrics ====
def fit(model, tr, va, epochs, lr, bs=64, patience=15, wd=1e-4):
    """Used by train.py only. tr/va = (inputs..., y)."""
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    T = [torch.as_tensor(t, dtype=torch.float32, device=dev) for t in tr]
    V = [torch.as_tensor(t, dtype=torch.float32, device=dev) for t in va]
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(T[-1]), device=dev)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            loss = F.mse_loss(model(*[t[b] for t in T[:-1]]), T[-1][b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = F.mse_loss(model(*V[:-1]), V[-1]).item()
        if v < best - 1e-5 or state is None:
            best, bad = v, 0
            state = {k: x.clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def predict_tensor(model, *xs):
    return model(*[torch.as_tensor(x, dtype=torch.float32, device=dev) for x in xs]).cpu().numpy()


def metrics(y, p, target="category"):
    """target='aqi': Acc = AQI-band accuracy. target='category': rounded 0-5 score accuracy."""
    acc = (aqi_band(p) == aqi_band(y)).mean() if target == "aqi" \
        else (np.clip(np.round(p), 0, 5) == y).mean()
    return dict(R2=r2_score(y, p) if len(y) > 1 else float("nan"),
                RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=mean_absolute_error(y, p), Acc=float(acc))