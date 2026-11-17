"""Ordered project presentation using measured, saved experiment artifacts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ecg_error_analysis import analyze_errors
from ecg_session3 import FEATURES, evaluate

MODELS = {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree"}


@st.cache_data(max_entries=2, show_spinner=False)
def load_results(root, signature):
    root = Path(root)
    source = root / "derived_session3"
    folder = root / "derived_classifier_comparison"
    manifest = pd.read_csv(source / "split_manifest.csv")
    beats = pd.read_csv(source / "beats.csv.gz")
    metadata = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
    assert not manifest.key.duplicated().any()
    train = manifest[manifest["split"] == "train"]
    test = manifest[manifest["split"] == "test"]
    assert set(train.key).isdisjoint(test.key)
    assert set(test.participant) <= set(train.participant)
    labels = sorted(train.participant.unique())
    results, summaries = {}, {}
    for unit in ("recording", "beat"):
        rows = []
        for slug, name in MODELS.items():
            predictions = pd.read_csv(folder / f"{slug}_{unit}_predictions.csv")
            assert set(predictions.recording_id) == set(test.key)
            keys = ["recording_id"] + (["beat_id"] if unit == "beat" else [])
            assert not predictions.duplicated(keys).any()
            cm = pd.read_csv(folder / f"{slug}_{unit}_confusion.csv", index_col=0)
            examples, directed, pairs = analyze_errors(predictions, cm)
            summary, _, _, _ = evaluate(predictions.person_id, predictions.predicted, labels, name, unit)
            rows.append([name, summary["accuracy"], summary["macro_precision"],
                         summary["macro_recall"], summary["macro_f1-score"]])
            results[(unit, slug)] = (cm, examples, directed, pairs)
        summaries[unit] = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
        # Ensure the two classifiers were evaluated on identical labeled samples.
        left = pd.read_csv(folder / f"logistic_regression_{unit}_predictions.csv")
        right = pd.read_csv(folder / f"decision_tree_{unit}_predictions.csv")
        pd.testing.assert_frame_equal(left[keys + ["person_id"]].sort_values(keys).reset_index(drop=True),
                                      right[keys + ["person_id"]].sort_values(keys).reset_index(drop=True))
    return manifest, beats, metadata, results, summaries


def render(root):
    st.title("HeartPrint — זיהוי אנשים לפי ECG")
    st.header("1. מטרת הפרויקט")
    st.write("לבדוק האם ניתן לזהות אדם מוכר לפי צורת פעימות ECG מהקלטה שלא שימשה לאימון, ולהשוות רגרסיה לוגיסטית לעץ החלטה.")
    source, folder = root / "derived_session3", root / "derived_classifier_comparison"
    paths = [source / "split_manifest.csv", source / "beats.csv.gz", folder / "run_metadata.json"]
    paths += [folder / f"{slug}_{unit}_{suffix}.csv" for slug in MODELS
              for unit in ("recording", "beat") for suffix in ("predictions", "confusion")]
    if not all(path.exists() for path in paths):
        st.info("להצגת תוצאות מחושבות יש להריץ תחילה ecg_session3.py ואז ecg_classifier_comparison.py.")
        return
    signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    manifest, beats, metadata, results, summaries = load_results(str(root), signature)

    st.header("2. מספר אנשים והקלטות")
    a, b, c = st.columns(3)
    a.metric("אנשים במאגר", manifest.participant.nunique())
    b.metric("הקלטות במאגר", len(manifest))
    c.metric("אנשים בניסוי", manifest.loc[manifest["split"] == "train", "participant"].nunique())
    st.caption(f"הקלטות מחוץ לניסוי: {int((manifest['split'] == 'excluded').sum())}. הספירות מחושבות ממיפוי ההקלטות השמור.")

    st.header("3. דוגמת ECG / beat")
    recording = st.selectbox("הקלטה לדוגמת פעימה", sorted(beats.recording_id.unique()))
    own = beats[beats.recording_id == recording]
    beat_id = st.selectbox("פעימה לדוגמה", own.beat_id.tolist())
    row = own[own.beat_id == beat_id].iloc[0]
    fig = go.Figure(go.Scatter(x=(np.arange(350) - 125) * 2,
                              y=row[FEATURES].to_numpy(dtype=float), name="פעימה מהמאגר"))
    fig.update_layout(xaxis_title="זמן ביחס לפסגת R (מילישניות)", yaxis_title="משרעת מנורמלת")
    st.plotly_chart(fig, width="stretch", alt="פעימת ECG אמיתית מתוך טבלת הפעימות השמורה")
    st.caption("פעימה מסוננת ומנורמלת מתוך ההקלטה שנבחרה, לא אות סינתטי. פסגות R זוהו אוטומטית ולא אומתו ידנית.")

    st.header("4. הסבר על train/test split")
    split_table = manifest.groupby("split").agg(recordings=("key", "size"), beats=("n_beats", "sum"))
    st.dataframe(split_table, alt="ספירת הקלטות ופעימות לפי קבוצת אימון, בדיקה והחרגות")
    st.write("החלוקה נעשית לפי recording_id: כל הפעימות מאותה הקלטה נשארות יחד. אין חפיפה בין הקלטות האימון והבדיקה, וכל אדם בבדיקה מיוצג באימון. שני המודלים משתמשים באותן דוגמאות בדיוק.")
    st.caption(f"Seed שמור: {metadata['seed']}. ה־StandardScaler מותאם רק לאימון. זו בדיקה על אנשים מוכרים ואינה מחייבת הפרדה בין ימי מדידה.")

    st.header("5. המודלים שנבדקו")
    st.write("Logistic Regression: למידה מ־350 ערכי הפעימה לאחר StandardScaler. Decision Tree Classifier: עץ החלטה על אותם ערכים, ללא צורך בהתאמת סקלות. מזהי אדם והקלטה אינם מאפייני קלט.")
    with st.expander("פרמטרים מההרצה השמורה"):
        st.json(metadata["parameters"])

    st.header("6. טבלת השוואת Accuracy, Precision, Recall, F1")
    unit = st.radio("יחידת ההערכה", ["recording", "beat"], horizontal=True,
                    format_func=lambda value: "הקלטה שלמה" if value == "recording" else "פעימה בודדת")
    comparison = summaries[unit]
    st.dataframe(comparison.style.format({c: "{:.2%}" for c in comparison.columns[1:]}),
                 hide_index=True, alt="מדדי שני המסווגים המחושבים מהתחזיות השמורות")
    st.caption("Precision, Recall ו־F1 בממוצע Macro: משקל שווה לכל אדם. Accuracy הוא שיעור התחזיות הנכונות. בהקלטה שלמה, שני המודלים ממוצעים הסתברויות על הפעימות. הטבלה מחושבת מהתחזיות השמורות ללא אימון מחדש.")
    st.download_button("הורדת טבלת ההשוואה", comparison.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"comparison_{unit}.csv", mime="text/csv")

    st.header("7. Confusion Matrix")
    for slug, name in MODELS.items():
        cm = results[(unit, slug)][0]
        fig = go.Figure(go.Heatmap(z=cm.to_numpy(), x=cm.columns.tolist(), y=cm.index.tolist(),
                                  colorscale="Blues", hovertemplate="אמיתי: %{y}<br>חזוי: %{x}<br>מספר: %{z}<extra></extra>"))
        fig.update_layout(title=name, xaxis_title="אדם חזוי", yaxis_title="אדם אמיתי", yaxis_autorange="reversed", height=550)
        st.plotly_chart(fig, width="stretch", alt=f"מטריצת בלבול מלאה של {name}")
    st.caption("שורות = אמת; עמודות = תחזית. האלכסון מציין תחזיות נכונות. כל המחלקות מוצגות; סולמות הצבע נפרדים.")

    st.header("8. דוגמאות לטעויות")
    for slug, name in MODELS.items():
        _, examples, _, pairs = results[(unit, slug)]
        st.subheader(name)
        st.write(f"מספר התחזיות השגויות: {len(examples)}")
        if examples.empty:
            st.info("אין טעויות ביחידת ההערכה שנבחרה.")
        else:
            st.dataframe(examples.head(20), hide_index=True, alt=f"דוגמאות true_person -> predicted_person עבור {name}")
            st.write("הזוגות המתבלבלים ביותר לפי סכום שני הכיוונים:")
            st.dataframe(pairs.head(10), hide_index=True, alt=f"דירוג זוגות מתבלבלים עבור {name}")
            st.caption("total = a_to_b + b_to_a; הספירות נבדקו מול מטריצת הבלבול. בתיקו מוצגים זוגות לפי סדר המזהים. מוצגות עד 20 דוגמאות ועד 10 זוגות.")
        st.download_button(f"הורדת כל הטעויות — {name}", examples.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{slug}_{unit}_error_examples.csv", mime="text/csv")

    st.header("9. מסקנה קצרה")
    best = comparison[comparison.Accuracy == comparison.Accuracy.max()]
    st.write(f"בחלוקה שנבדקה, ה־Accuracy הגבוה ביותר הוא {best.Accuracy.iloc[0]:.2%}, של "
             f"{', '.join(best.Model)}. הפער בין המודלים הוא "
             f"{100 * (comparison.Accuracy.max() - comparison.Accuracy.min()):.2f} נקודות אחוז.")
    st.caption("זו חלוקה קבועה יחידה של אנשים מוכרים. אין להסיק ממנה ביצועים על אנשים חדשים או ביום מדידה אחר. פעימות מאותה הקלטה אינן דוגמאות עצמאיות.")
