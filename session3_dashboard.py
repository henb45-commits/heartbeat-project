"""Display measured session-3 results without retraining or reading raw signals."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NAMES = {"baseline": "Baseline — תבנית ממוצעת", "logistic": "Logistic Regression"}
COLORS = {"baseline": "#2687cf", "logistic": "#209d91"}
_source_stat = Path(__file__).stat()
SOURCE_SIGNATURE = (_source_stat.st_mtime_ns, _source_stat.st_size)


@st.cache_data(max_entries=2, show_spinner=False)
def load_beat_table(path, signature):
    return pd.read_csv(path)


def render_beat_table(root):
    st.title("פירוק ההקלטות של כל אדם לפעימות")
    path = root / "derived_session3" / "beats.csv.gz"
    if not path.exists():
        st.info("יש להריץ תחילה את ecg_session3.py כדי להפיק את טבלת הפעימות.")
        return
    beats = load_beat_table(path, (path.stat().st_size, path.stat().st_mtime_ns))
    person = st.selectbox("בחרי אדם", sorted(beats.person_id.unique()), key="table_person")
    person_beats = beats[beats.person_id == person]
    manifest = pd.read_csv(path.parent / "split_manifest.csv").fillna("")
    records = manifest[manifest.participant == person].copy()
    records = records.rename(columns={"key": "הקלטה", "n_beats": "פעימות שימושיות", "split": "קבוצה", "error": "הערה"})
    records["קבוצה"] = records["קבוצה"].map({"train": "אימון", "test": "בדיקה", "excluded": "מחוץ לניסוי"})
    st.subheader("ההקלטות של האדם שנבחר")
    st.dataframe(records[["הקלטה", "פעימות שימושיות", "קבוצה", "הערה"]], hide_index=True,
                 width="stretch", alt="כל ההקלטות של האדם, מספר הפעימות וחלוקת האימון והבדיקה")
    recording = st.selectbox("בחרי הקלטה לפירוק", ["כל ההקלטות"] + records["הקלטה"].tolist(), key="table_recording")
    selected = person_beats if recording == "כל ההקלטות" else person_beats[person_beats.recording_id == recording]
    st.subheader("כל שורה היא פעימת לב אחת")
    st.caption(f"{person} · {len(selected):,} פעימות בטבלה · 350 נקודות זמן לכל פעימה. "
               "אפשר לגלול את הטבלה ימינה ושמאלה כדי לראות את כל נקודות הזמן.")
    st.info("T1 = 250 מילישניות לפני פסגת R; T126 = פסגת R; T350 = 448 מילישניות אחריה. "
            "המרווח בין העמודות הוא 2 מילישניות. הערכים בתאים הם עוצמת האות המסוננת והמנורמלת.")
    if selected.empty:
        st.info("בהקלטה זו לא חולצו פעימות שעברו את תנאי האיכות הקיימים; הסיבה מופיעה בטבלת ההקלטות למעלה.")
        return
    st.subheader("הטבלה כמו בשקף 4 במצגת")
    st.caption("כל שורה היא פעימה אחת: הצורה שלה, ערכי האות בנקודות זמן, ובסוף תווית האדם. הגרף הקטן בכל שורה מכיל את כל 350 הנקודות.")
    slide_table = selected[["recording_id", "beat_id"]].copy()
    slide_table["צורת הפעימה — 350 נקודות"] = selected[[f"T{i}" for i in range(1,351)]].to_numpy().tolist()
    for i in [1, 50, 100, 125, 126, 127, 150, 200, 250, 300, 350]:
        slide_table[f"T{i} ({2*(i-126):+d} ms)"] = selected[f"T{i}"]
    slide_table["תווית — מי האדם"] = selected.person_id
    slide_config = {
        "recording_id": st.column_config.TextColumn("הקלטה"),
        "beat_id": st.column_config.NumberColumn("פעימה — מיקום R", format="%d"),
        "צורת הפעימה — 350 נקודות": st.column_config.LineChartColumn("צורת הפעימה — 350 נקודות", width="large"),
        "תווית — מי האדם": st.column_config.TextColumn("תווית — מי האדם"),
    }
    slide_config.update({c: st.column_config.NumberColumn(c, format="%.3f") for c in slide_table.columns if c.startswith("T")})
    st.dataframe(slide_table, column_config=slide_config, hide_index=True, row_height=75,
                 height=550, width="stretch", alt="טבלה כמו במצגת: גרף פעימה בכל שורה, נקודות הזמן ותווית האדם")
    st.caption("בטבלה זו מוצגות 11 עמודות זמן לדוגמה כדי להקל על הקריאה. כל 350 העמודות מופיעות בטבלה המלאה שמתחתיה.")
    st.subheader("הטבלה המלאה — כל העמודות T1 עד T350")
    display_table = selected.copy()
    display_table.insert(3, "R_time_seconds", display_table.beat_id / 500)
    display_table.insert(4, "beat_number", display_table.groupby("recording_id").cumcount() + 1)
    config = {
        "person_id": st.column_config.TextColumn("אדם", pinned=True),
        "recording_id": st.column_config.TextColumn("הקלטה", pinned=True),
        "beat_id": st.column_config.NumberColumn("beat_id — דגימת R", format="%d"),
        "beat_number": st.column_config.NumberColumn("מספר פעימה בהקלטה", format="%d"),
        "R_time_seconds": st.column_config.NumberColumn("זמן R מתחילת ההקלטה (שניות)", format="%.3f"),
    }
    for i in range(1,351):
        ms = 2*(i-126)
        config[f"T{i}"] = st.column_config.NumberColumn(f"T{i} ({ms:+d} ms)", format="%.4f",
                                                        help=f"עוצמת האות המנורמלת בזמן {ms:+d} מילישניות ביחס ל־R")
    st.dataframe(display_table, column_config=config, hide_index=True, width="stretch", height=550,
                 alt="פירוק ההקלטות הנבחרות: שורה לכל פעימה ועמודה לכל אחת מ־350 נקודות הזמן")
    st.caption("beat_id הוא מיקום פסגת R בדגימות בהקלטה המקורית; זמן R הוא אותו מיקום בשניות. "
               "מספר הפעימה הוא הסדר בין הפעימות שנשמרו אחרי סינון. הפסגות זוהו אוטומטית ולא אומתו ידנית.")
    st.subheader("מהצורה של הפעימה אל המספרים בשורה — כמו בשקף 4")
    illustration_record = selected.recording_id.iloc[0]
    example = selected[selected.recording_id == illustration_record].head(3)
    st.caption(f"שלוש הפעימות השימושיות הראשונות מתוך {illustration_record}. כל קו בגרף הוא שורה אחת בטבלה שמתחתיו.")
    fig = go.Figure()
    time_ms = [2*(i-126) for i in range(1,351)]
    for _, row in example.iterrows():
        fig.add_trace(go.Scatter(x=time_ms, y=row[[f"T{i}" for i in range(1,351)]].to_numpy(dtype=float),
                                name=f"beat_id={int(row.beat_id)}", mode="lines"))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title="זמן ביחס ל־R (מילישניות)", yaxis_title="עוצמת אות מנורמלת")
    plot(fig, "שלוש פעימות אמיתיות מיושרות סביב R")
    sample_columns = ["person_id", "recording_id", "beat_id", "T1", "T125", "T126", "T127", "T200", "T350"]
    st.dataframe(example[sample_columns].rename(columns={f"T{i}": f"T{i} ({2*(i-126):+d} ms)" for i in [1,125,126,127,200,350]}),
                 hide_index=True, width="stretch", alt="אותן שלוש פעימות כשורות מספריות, עם מבחר נקודות זמן")
    st.caption("הטבלה הקצרה מדגימה כמה עמודות; הטבלה המלאה למעלה והקובץ להורדה מכילים את כל 350 הנקודות.")
    st.download_button("הורדת טבלת הפעימות המוצגת — כל 350 הנקודות", display_table.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{person}_beats.csv", mime="text/csv")


def render_slide_walkthrough(root):
    from session3_story import render
    render(root)


def render_slide_walkthrough_legacy(root):
    st.title("מתיאור לזיהוי — לפי מצגת מפגש 3")
    st.caption("מבוסס על שמונת השקפים של ׳מפגש 3 - מתיאור לזיהוי׳. הנתונים והגרפים כאן מחושבים מקובצי הפרויקט.")
    stage = st.selectbox("בחרי צעד מהמצגת", ["1 — מה אפשר לבדוק", "2 — מהקלטה לטבלה", "3 — מה מסווגים", "4 — בדיקה הוגנת", "5 — כמה נכון ואיפה טועים"], index=1)
    folder = root / "derived_session3"
    if not (folder / "beats.csv.gz").exists():
        st.info("יש להריץ תחילה את ecg_session3.py.")
        return
    if stage.startswith("2"):
        st.markdown("**לפי שקף 4:** חלון סביב כל שיא R הופך לשורה; כל נקודת זמן הופכת לעמודה; person_id היא התווית שאותה לומדים לזהות.")
        st.caption("הבחירות בניסוי: ערוץ מסונן, חלון ‎−250 עד ‎+450 מילישניות, נרמול כל פעימה — הדגש הוא על צורת הפעימה ולא על הגובה המוחלט.")
        render_beat_table(root)
        return
    manifest = pd.read_csv(folder / "split_manifest.csv").fillna("")
    if stage.startswith("1"):
        rows = []
        for person, group in manifest.groupby("participant"):
            dates = set()
            for key in group.key:
                header = root / "ecg-id-database-1.0.0" / (key + ".hea")
                for line in header.read_text(encoding="utf-8").splitlines():
                    if line.startswith("# ECG date:"):
                        dates.add(line.split(":",1)[1].strip())
            rows.append({"אדם": person, "הקלטות": len(group), "הקלטות שימושיות": int((group.n_beats>0).sum()), "תאריכי מדידה שונים": len(dates)})
        people = pd.DataFrame(rows)
        st.metric("אנשים עם הקלטות ביותר מיום אחד", int((people["תאריכי מדידה שונים"]>1).sum()))
        st.dataframe(people, hide_index=True, alt="מספר הקלטות ותאריכי מדידה לכל אדם")
        st.info("אפשר לבדוק זיהוי מהקלטה נפרדת. החלוקה הנוכחית אינה דורשת יום אחר, ולכן אין להסיק ממנה לבדה זיהוי לאורך זמן או תחת לחץ.")
    elif stage.startswith("3"):
        st.subheader("קלט: פעימה אחת · פלט: מזהה האדם")
        st.markdown("המודל מקבל את T1…T350. הוא אינו מקבל את person_id או recording_id כמאפיינים. person_id היא התשובה הנכונה בזמן האימון.")
        st.table(pd.DataFrame({"שיטה": ["תבנית ממוצעת", "רגרסיה לוגיסטית"], "איך מתקבלת החלטה": ["משווים את צורת הפעימה לתבניות שנבנו מהאימון", "לומדים משקל לכל נקודת זמן ולכל מחלקת אדם"]}), alt="הבדל בין שיטת התבניות למודל הלומד")
        st.info("המצגת מסבירה גם עץ החלטה ויער אקראי כחלופות. בפרויקט אומנה רגרסיה לוגיסטית בהתאם למשימת מפגש 3. בפועל משתתפים בניסוי 89 מתוך 90 האנשים, משום שלאדם אחד יש רק הקלטה אחת.")
        render_scatter(root, manifest)
    elif stage.startswith("4"):
        st.subheader("הקלטה שלמה נשארת בצד אחד")
        summary = manifest.groupby("split").agg(recordings=("key","size"), beats=("n_beats","sum")).rename(index={"train":"אימון", "test":"בדיקה", "excluded":"מחוץ לניסוי"})
        st.dataframe(summary.rename(columns={"recordings":"הקלטות", "beats":"פעימות"}), alt="גודל קבוצות האימון והבדיקה לפי הקלטות ופעימות")
        st.dataframe(manifest[["participant","key","split","n_beats"]], hide_index=True, alt="פירוט שיוך כל הקלטה לאימון או לבדיקה")
        st.info("כל הפעימות מאותה הקלטה נשארות יחד. לכל אדם בבדיקה יש הקלטת אימון. ה־scaler והמודל לומדים רק מהאימון. בדיקה ביום אחר דורשת ניסוי נפרד.")
    else:
        st.subheader("כל פעימה מצביעה — הרוב קובע, כמו בשקף 8")
        predictions = pd.read_csv(folder / "beat_predictions.csv")
        votes = predictions.groupby(["person_id", "recording_id"]).agg(
            baseline=("baseline", lambda x: x.value_counts().loc[lambda s:s==s.max()].index.sort_values()[0]),
            logistic=("logistic", lambda x: x.value_counts().loc[lambda s:s==s.max()].index.sort_values()[0])) .reset_index()
        st.caption("בהצבעה כל פעימה תורמת קול אחד. בתיקו נבחר המזהה הראשון בסדר אלפביתי, בכלל קבוע מראש. זו השוואה נוספת; התוצאה המקורית משתמשת בתבנית הקלטה ובממוצע הסתברויות.")
        summaries = []
        for method in NAMES:
            correct = int((votes[method] == votes.person_id).sum())
            summaries.append({"שיטה": NAMES[method], "נכון":correct, "טעויות":len(votes)-correct, "דיוק":correct/len(votes)})
        st.dataframe(pd.DataFrame(summaries).style.format({"דיוק":"{:.2%}"}), hide_index=True, alt="תוצאות זיהוי הקלטה לפי רוב קולות הפעימות")
        st.dataframe(votes, hide_index=True, alt="האדם האמיתי וניבוי רוב הקולות לכל הקלטה")
        st.markdown("**גרפי Precision/Recall/F1, מטריצות בלבול ופירוט הטעויות של הניסוי המקורי זמינים בתפריט ׳מפגש 3 — השוואת מודלים׳.**")


def render_scatter(root, manifest):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    st.subheader("גרף פיזור וקו הפרדה — כמו בשקף 6")
    st.caption("כל נקודה היא פעימה אמיתית. הצבע מציין את האדם; עיגול = אימון, X = בדיקה. שני הצירים הם עוצמת האות בשתי נקודות זמן לבחירתך.")
    path = root / "derived_session3" / "beats.csv.gz"
    beats = load_beat_table(path, (path.stat().st_size, path.stat().st_mtime_ns))
    people = sorted(manifest.loc[manifest.split == "train", "participant"].unique())
    left, right = st.columns(2)
    first = left.selectbox("אדם א׳", people, key="scatter_person_a")
    second = right.selectbox("אדם ב׳", [p for p in people if p != first], key="scatter_person_b")
    columns = [f"T{i}" for i in range(1,351)]
    def time_label(column):
        return f"{column} — {2*(int(column[1:])-126):+d} ms ביחס ל־R"
    x = left.selectbox("נקודת זמן לציר X", columns, index=125, format_func=time_label, key="scatter_x")
    remaining = [c for c in columns if c != x]
    y = right.selectbox("נקודת זמן לציר Y", remaining, index=remaining.index("T200") if "T200" in remaining else 0,
                        format_func=time_label, key="scatter_y")
    points = beats[beats.person_id.isin([first, second])].copy()
    points["split"] = points.recording_id.map(manifest.set_index("key").split)
    points = points[points.split.isin(["train", "test"])]
    train = points[points.split == "train"]
    test = points[points.split == "test"]
    model = Pipeline([("scaler", StandardScaler()),
                      ("logistic", LogisticRegression(solver="lbfgs", C=1, random_state=42, max_iter=3000))])
    model.fit(train[[x,y]], train.person_id)
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["logistic"]
    weights = classifier.coef_[0] / scaler.scale_
    intercept = classifier.intercept_[0] - weights @ scaler.mean_
    fig = go.Figure()
    # Plot the zero-score boundary in original feature units. This uses only
    # training-fitted coefficients; displaying test points does not fit the model.
    ranges = []
    for column in [x,y]:
        low, high = points[column].min(), points[column].max()
        pad = max((high-low)*.08, .01)
        ranges.append(np.linspace(low-pad, high+pad, 100))
    grid_x, grid_y = np.meshgrid(*ranges)
    scores = weights[0]*grid_x + weights[1]*grid_y + intercept
    if scores.min() <= 0 <= scores.max():
        fig.add_trace(go.Contour(x=ranges[0], y=ranges[1], z=scores,
                                contours=dict(start=0, end=0, size=1, coloring="lines"),
                                line=dict(color="#18304b", width=3), showscale=False,
                                hoverinfo="skip", name="קו הפרדה שנלמד באימון"))
    for person, color in [(first, "#209d91"), (second, "#2687cf")]:
        for split, symbol, label in [("train", "circle", "אימון"), ("test", "x", "בדיקה")]:
            group = points[(points.person_id == person) & (points.split == split)]
            fig.add_trace(go.Scatter(x=group[x], y=group[y], mode="markers", name=f"{person} · {label}",
                                    marker=dict(color=color, symbol=symbol, size=8, opacity=.7),
                                    customdata=group[["recording_id","beat_id"]].to_numpy(),
                                    hovertemplate="%{customdata[0]}<br>beat_id=%{customdata[1]}<br>X=%{x:.3f}<br>Y=%{y:.3f}<extra>%{fullData.name}</extra>"))
    fig.update_layout(xaxis_title=f"עוצמה מנורמלת ב־{time_label(x)}", yaxis_title=f"עוצמה מנורמלת ב־{time_label(y)}")
    plot(fig, "פיזור פעימות של שני אנשים בשתי נקודות זמן וקו הפרדה שנלמד מהאימון", 620)
    correct = int((model.predict(test[[x,y]]) == test.person_id).sum())
    st.metric("דיוק מודל ההמחשה על פעימות הבדיקה של שני האנשים", f"{correct/len(test):.2%}")
    st.caption(f"{correct} מתוך {len(test)} פעימות בדיקה זוהו נכון. הקו וה־scaler נלמדו רק מ־{len(train)} פעימות אימון, לפי חלוקת ההקלטות המקורית.")
    st.info("זהו מודל המחשה נפרד: שני אנשים ושתי עמודות בלבד, כפי שמאויר במצגת. המודל הראשי משתמש ב־350 עמודות וב־89 אנשים. בחירת נקודות זמן שונות עשויה להציג חפיפה בין האנשים; הנתונים אינם מבטיחים קו שמפריד באופן מושלם. הסתכלות בגרף אינה משנה את תוצאות המודל הראשי.")


def plot(fig, name, height=420):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=45,r=25,t=45,b=60),
                      font=dict(family="Arial", size=14), legend=dict(orientation="h", y=-.2))
    st.plotly_chart(fig, width="stretch", alt=name, config={"displaylogo": False})


@st.cache_data(max_entries=2, show_spinner=False)
def load_results(folder, signature):
    metrics = pd.read_csv(folder / "metrics.csv")
    metadata = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
    details = {}
    for unit in ("recording", "beat"):
        for method in NAMES:
            prefix = f"{unit}_{method}"
            details[(unit, method)] = {
                "confusion": pd.read_csv(folder / f"{prefix}_confusion.csv", index_col=0),
                "errors": pd.read_csv(folder / f"{prefix}_errors.csv"),
                "report": pd.read_csv(folder / f"{prefix}_classification_report.csv", index_col=0),
            }
    return metrics, metadata, details


def render_comparison(root):
    st.title("מפגש 3 — איזו שיטה מזהה טוב יותר?")
    st.caption("תוצאות ההרצה בפועל · אותן הקלטות בדיקה לשתי השיטות · ללא אימון מחדש בפתיחת הדשבורד")
    folder = root / "derived_session3"
    if not (folder / "metrics.csv").exists():
        st.info("יש להריץ תחילה את ecg_session3.py כדי להפיק את תוצאות ההשוואה.")
        return
    files = sorted(folder.glob("*.csv")) + [folder / "run_metadata.json"]
    signature = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    metrics, metadata, details = load_results(folder, signature)
    level = st.segmented_control("יחידת הזיהוי", ["הקלטה שלמה", "פעימה בודדת"], default="הקלטה שלמה")
    unit = "beat" if level == "פעימה בודדת" else "recording"
    selected = metrics[metrics.unit == unit].set_index("method")
    base, logistic = selected.loc["baseline"], selected.loc["logistic"]
    a,b,c = st.columns(3)
    a.metric("דיוק Baseline", f"{base.accuracy:.2%}", help=f"{int(base.correct):,} מתוך {int(base.total):,}")
    b.metric("דיוק Logistic Regression", f"{logistic.accuracy:.2%}", help=f"{int(logistic.correct):,} מתוך {int(logistic.total):,}")
    c.metric("פער הדיוק", f"{100*(logistic.accuracy-base.accuracy):+.2f} נקודות אחוז")
    st.caption(f"באימון: {metadata['train_recordings']} הקלטות ו־{metadata['train_beats']:,} פעימות. "
               f"בבדיקה: {metadata['test_recordings']} הקלטות ו־{metadata['test_beats']:,} פעימות. "
               f"{metadata['classes']} אנשים משתתפים בניסוי.")
    if unit == "recording":
        st.info("כל החלטה מתייחסת להקלטה שלמה. ברגרסיה הלוגיסטית ממוצעים את הסתברויות הפעימות בכל הקלטה. זו ההשוואה ל־88 מתוך 102.")
    else:
        st.info("כל החלטה מתייחסת לפעימה אחת. ה־baseline משווה כל פעימה לגלריית התבניות הקיימת. אין להשוות ישירות את אחוז הפעימות לאחוז ההקלטות.")

    left,right = st.columns(2)
    with left:
        st.subheader("אחוז הזיהויים הנכונים")
        fig = go.Figure()
        for method, row in selected.iterrows():
            fig.add_trace(go.Bar(x=[NAMES[method]], y=[100*row.accuracy],
                                marker_color=COLORS[method], text=[f"{row.accuracy:.2%}"],
                                textposition="outside", showlegend=False))
        fig.update_yaxes(range=[0,105], title="דיוק (%)")
        plot(fig, "השוואת אחוז הזיהוי הנכון בין שתי השיטות")
    with right:
        st.subheader("כמה הצלחות וכמה טעויות?")
        fig = go.Figure()
        for column, label, color in (("correct", "נכון", "#209d91"), ("errors", "שגוי", "#dc6477")):
            values = selected[column].astype(int).tolist()
            fig.add_trace(go.Bar(x=[NAMES[m] for m in selected.index], y=values,
                                name=label, marker_color=color, text=values, textposition="auto"))
        fig.update_layout(barmode="stack", yaxis_title="מספר זיהויים")
        plot(fig, "מספר הזיהויים הנכונים והשגויים בכל שיטה")

    st.subheader("טבלת השוואה מלאה")
    average = st.segmented_control("אופן חישוב Precision / Recall / F1", ["Macro — משקל שווה לכל אדם", "Weighted — לפי מספר דוגמאות"], default="Macro — משקל שווה לכל אדם")
    prefix = "weighted" if average and average.startswith("Weighted") else "macro"
    comparison = selected[["total", "correct", "errors", "accuracy", f"{prefix}_precision", f"{prefix}_recall", f"{prefix}_f1-score"]].copy()
    comparison.index = comparison.index.map(NAMES)
    comparison.columns = ["נבדקו", "נכון", "טעויות", "Accuracy", "Precision", "Recall", "F1"]
    st.dataframe(comparison.style.format({c: "{:.2%}" for c in ["Accuracy", "Precision", "Recall", "F1"]}), width="stretch", alt="טבלת מדדי ההשוואה לשתי השיטות")
    fig = go.Figure()
    for method, row in selected.iterrows():
        fig.add_trace(go.Bar(name=NAMES[method], x=["Precision", "Recall", "F1"],
                            y=[100*row[f"{prefix}_{k}"] for k in ["precision", "recall", "f1-score"]],
                            marker_color=COLORS[method]))
    fig.update_layout(barmode="group", yaxis_title="אחוזים")
    fig.update_yaxes(range=[0,100])
    plot(fig, "השוואת Precision, Recall ו־F1")
    st.caption("Precision: כמה מהשיוכים לאדם היו נכונים. Recall: כמה מהדוגמאות שלו זוהו. F1: שילוב של שניהם. מדד ללא ניבויים מוגדר כ־0.")

    st.subheader("מטריצות בלבול — מי זוהה בתור מי?")
    show_all = st.checkbox("הצגת כל האנשים במטריצות", value=False)
    involved = sorted({p for method in NAMES for column in ["true", "predicted"]
                       for p in details[(unit, method)]["errors"][column]})
    for col, method in zip(st.columns(2), NAMES):
        with col:
            cm = details[(unit, method)]["confusion"]
            if not show_all and involved:
                cm = cm.loc[involved, involved]
            fig = go.Figure(go.Heatmap(z=cm.to_numpy(), x=cm.columns.tolist(), y=cm.index.tolist(),
                                      colorscale="Blues", hovertemplate="אמיתי: %{y}<br>חזוי: %{x}<br>מספר: %{z}<extra></extra>"))
            fig.update_layout(title=NAMES[method], xaxis_title="אדם חזוי", yaxis_title="אדם אמיתי", yaxis_autorange="reversed")
            plot(fig, f"מטריצת בלבול של {NAMES[method]}", 570)
    st.caption("שורות = האדם האמיתי; עמודות = האדם החזוי. האלכסון הוא זיהוי נכון. ברירת המחדל מציגה אותה קבוצת אנשים המעורבים בטעות באחת השיטות; כל מטריצה משתמשת בסולם צבעים משלה.")

    st.subheader("הטעויות הנפוצות")
    for col, method in zip(st.columns(2), NAMES):
        with col:
            st.markdown(f"**{NAMES[method]}**")
            errors = details[(unit, method)]["errors"]
            top = errors.head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["count"], y=top.true + " → " + top.predicted,
                                   orientation="h", marker_color=COLORS[method], text=top["count"], textposition="auto"))
            fig.update_layout(xaxis_title="מספר טעויות")
            plot(fig, f"עשרת זוגות השיוך השגוי השכיחים של {NAMES[method]}")
            st.dataframe(errors.rename(columns={"true":"אדם אמיתי", "predicted":"זוהה בתור", "count":"מספר טעויות"}), hide_index=True, alt="כל זוגות הזיהוי השגויים ומספר הופעותיהם")
    counts = {m: int(details[(unit,m)]["confusion"].loc["Person_52", "Person_64"]) for m in NAMES}
    st.info(f"Person_52 ← מזוהה בתור Person_64: ב־Baseline {counts['baseline']} פעמים; ברגרסיה הלוגיסטית {counts['logistic']} פעמים. יחידת הספירה: {level}.")

    st.subheader("מדדים לכל אדם")
    method = st.selectbox("שיטה להצגת הטבלה המפורטת", list(NAMES), format_func=NAMES.get)
    report = details[(unit, method)]["report"].copy()
    st.dataframe(report.style.format(precision=3), width="stretch", alt="Precision, Recall, F1 ומספר דוגמאות לכל אדם")
    st.download_button("הורדת טבלת ההשוואה כ־CSV", comparison.to_csv().encode("utf-8-sig"), file_name=f"ecg_{unit}_comparison.csv", mime="text/csv")
    st.caption("כל הקלטה נמצאת רק באימון או בבדיקה. ה־scaler והמודל לומדים רק מהאימון. ההפרדה אינה בהכרח בין תאריכים; אין כאן הוכחה לזיהוי תחת לחץ או של אדם לא מוכר.")
