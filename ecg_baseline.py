"""Read-only ECG-ID analysis, matching heartbeat_analysis.ipynb."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt, find_peaks

SEED = 42
PRE_S, POST_S = 0.25, 0.45
MIN_BEATS = 4

def normalize_template(values):
    values = np.asarray(values, dtype=float)
    values = values - values.mean()
    length = np.linalg.norm(values)
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError("Degenerate template")
    return values / length

def extract_beats(record_path):
    record = wfdb.rdrecord(str(record_path), physical=True)
    fs = float(record.fs)
    if "ECG I filtered" not in record.sig_name:
        raise ValueError("Expected filtered ECG channel not found")
    channel = record.sig_name.index("ECG I filtered")
    signal = record.p_signal[:, channel].astype(float)
    if not np.isfinite(signal).all():
        raise ValueError("Missing/nonfinite signal samples; no imputation applied")
    # רוחב פס לצורת הגל; מסלול נפרד להדגשת QRS לצורך איתור R בלבד
    clean = sosfiltfilt(butter(3, [0.5, 40], btype="bandpass", fs=fs, output="sos"), signal)
    qrs = sosfiltfilt(butter(3, [5, 20], btype="bandpass", fs=fs, output="sos"), signal)
    energy = np.convolve(qrs ** 2, np.ones(max(1, round(0.08*fs))) / max(1, round(0.08*fs)), mode="same")
    candidates, _ = find_peaks(energy, distance=round(0.35*fs),
                               prominence=max(0.35*np.std(energy), 1e-12))
    radius = round(0.08*fs)
    peaks = []
    for peak in candidates:
        left, right = max(0, peak-radius), min(len(qrs), peak+radius+1)
        aligned = left + int(np.argmax(np.abs(qrs[left:right])))
        if not peaks or aligned - peaks[-1] >= round(0.30*fs):
            peaks.append(aligned)
    peaks = np.asarray(peaks, dtype=int)
    pre, post = round(PRE_S*fs), round(POST_S*fs)
    valid = peaks[(peaks >= pre) & (peaks+post <= len(clean))]
    beats = np.asarray([clean[p-pre:p+post] for p in valid])
    if len(beats) < MIN_BEATS:
        raise ValueError(f"Only {len(beats)} complete beats")
    # לכל פעימה מסירים היסט ומנרמלים סקלה; אין שימוש בזהויות אחרות
    centered = beats - beats.mean(axis=1, keepdims=True)
    scales = centered.std(axis=1)
    keep = scales > 1e-10
    median_scale = np.median(scales[keep]) if keep.any() else 0
    keep &= (scales >= 0.25*median_scale) & (scales <= 4*median_scale)
    beats = centered[keep] / scales[keep, None]
    valid = valid[keep]
    if len(beats) < MIN_BEATS:
        raise ValueError("Too few nondegenerate beats after amplitude screening")
    mean_beat = beats.mean(axis=0)
    return dict(fs=fs, signal=signal, clean=clean, peaks=peaks, used_peaks=valid,
                beats=beats, mean_beat=mean_beat, vector=normalize_template(mean_beat),
                time=np.arange(-pre, post)/fs)

def analyze(data_root):
    data_root = Path(data_root)
    headers = sorted(data_root.glob("Person_*/*.hea"))
    if not headers:
        raise ValueError("לא נמצאו קובצי HEA בתיקיית המאגר.")
    beats, rows = {}, []
    for path in headers:
        lines = path.read_text(encoding="utf-8").splitlines()
        first = lines[0].split()
        meta = dict(line[1:].strip().split(":", 1) for line in lines if line.startswith("#") and ":" in line)
        key = f"{path.parent.name}/{path.stem}"
        row = dict(key=key, participant=path.parent.name, record=path.stem,
                   fs=float(first[2]), duration=int(first[3])/float(first[2]),
                   age=pd.to_numeric(meta.get("Age", "").strip(), errors="coerce"),
                   sex=meta.get("Sex", "").strip(),
                   date=pd.to_datetime(meta.get("ECG date", "").strip(),format="%d.%m.%Y",errors="coerce"),
                   n_beats=0, error="")
        try:
            data = extract_beats(path.with_suffix(""))
            beats[key] = data
            row["n_beats"] = len(data["beats"])
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    recordings = pd.DataFrame(rows)
    good = recordings[recordings.error == ""]
    counts = good.groupby("participant").size()
    labels = sorted(counts[counts >= 2].index)
    if len(labels) < 2:
        raise ValueError("נדרשים לפחות שני משתתפים עם שתי הקלטות שימושיות.")
    rng = np.random.default_rng(SEED)
    train, test = [], []
    for person in labels:
        keys = sorted(good.loc[good.participant == person, "key"])
        keys = rng.permutation(keys).tolist()
        size = min(len(keys)-1,max(1,math.ceil(.2*len(keys))))
        test.extend(keys[:size]); train.extend(keys[size:])
    assert not set(train) & set(test)
    if len({len(beats[k]["vector"]) for k in train+test}) != 1:
        raise ValueError("תדרי הדגימה אינם אחידים; נדרשת התאמה מפורשת.")
    gallery = {p:normalize_template(np.mean([beats[k]["vector"] for k in train if k.split("/")[0] == p],axis=0)) for p in labels}
    matrix = np.vstack([gallery[p] for p in labels])
    scores = np.clip(np.vstack([beats[k]["vector"] for k in test]) @ matrix.T,-1,1)
    truth = [k.split("/")[0] for k in test]
    indices = np.array([labels.index(p) for p in truth])
    own = scores[np.arange(len(test)),indices]
    other = scores.copy(); other[np.arange(len(test)),indices] = -np.inf
    results = pd.DataFrame(dict(recording=test,true=truth,predicted=[labels[i] for i in scores.argmax(axis=1)],self_similarity=own,best_other_similarity=other.max(axis=1)))
    results["correct"] = results.true == results.predicted
    confusion = pd.crosstab(results.true,results.predicted).reindex(index=labels,columns=labels,fill_value=0)
    return dict(recordings=recordings,beats=beats,labels=labels,train=train,test=test,gallery=gallery,
                gallery_matrix=matrix,scores=scores,results=results,confusion=confusion)
