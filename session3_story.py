"""Seven ordered, data-backed teaching sections for meeting 3 only."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

FEATURES = [f"T{i}" for i in range(1,351)]
COLORS = ["#209d91", "#2687cf"]


@st.cache_data(max_entries=2, show_spinner=False)
def load(folder, signature):
    beats = pd.read_csv(folder / "beats.csv.gz")
    manifest = pd.read_csv(folder / "split_manifest.csv").fillna("")
    with np.load(folder / "visualization_model.npz", allow_pickle=False) as saved:
        model = {k:saved[k] for k in saved.files}
    assert str(model["source_digest"]) == hashlib.sha256((folder / "beats.csv.gz").read_bytes()).hexdigest(), "יש להפיק מחדש את visualization_model.npz לאחר שינוי טבלת הפעימות"
    assert len(model["coordinates"]) == len(beats)
    return beats, manifest, model


def chart(fig, name, height=430):
    fig.update_layout(template="plotly_white", height=height, font=dict(family="Arial",size=14),
                      margin=dict(l=40,r=25,t=40,b=60), legend=dict(orientation="h",y=-.23))
    st.plotly_chart(fig, width="stretch", alt=name, config={"displaylogo":False})


def render(root):
    folder = root / "derived_session3"
    st.title("מפגש 3 — מתיאור לזיהוי")
    st.caption("מה אפשר לבדוק ← מהקלטה לטבלה ← מה המודל לומד ← המחשה ← בדיקה הוגנת ← התוצאות שלי")
    paths = [folder / n for n in ["beats.csv.gz","split_manifest.csv","visualization_model.npz"]]
    if not all(p.exists() for p in paths):
        st.info("להכנת גרפי המשקלים ו־PCA יש להריץ את ecg_session3_visuals.py לאחר ecg_session3.py.")
        return
    beats, manifest, model = load(folder, tuple((p.name,p.stat().st_mtime_ns,p.stat().st_size) for p in paths))
    split = beats.recording_id.map(manifest.set_index("key").split)
    train_ids = set(manifest.loc[manifest.split=="train","key"])
    test_ids = set(manifest.loc[manifest.split=="test","key"])
    assert train_ids.isdisjoint(test_ids)
    assert set(beats.loc[split=="test","person_id"]) <= set(beats.loc[split=="train","person_id"])

    st.header("1. מה אפשר לבדוק")
    people = manifest.groupby("participant").agg(recordings=("key","size"),usable=("n_beats",lambda x:int((x>0).sum()))).reset_index()
    a,b=st.columns(2)
    a.metric("אנשים במאגר",len(people))
    b.metric("אנשים עם יותר מהקלטה אחת",int((people.recordings>1).sum()))
    fig=go.Figure(go.Bar(x=people.participant,y=people.recordings,marker_color=COLORS[0]))
    fig.update_layout(xaxis_title="אדם",yaxis_title="מספר הקלטות")
    chart(fig,"מספר ההקלטות האמיתי לכל אדם")
    st.dataframe(people.rename(columns={"participant":"אדם","recordings":"הקלטות","usable":"הקלטות עם פעימות שימושיות"}),hide_index=True,alt="מספר הקלטות לכל אדם")

    st.header("2. מהקלטה לטבלה")
    person=st.selectbox("אדם לדוגמת הטבלה",sorted(beats.person_id.unique()),key="story_table_person")
    own=beats[beats.person_id==person]
    recording=st.selectbox("הקלטה לדוגמה",own.recording_id.unique().tolist(),key="story_record")
    rows=own[own.recording_id==recording]
    beat_id=st.selectbox("פעימה להצגה בגרף",rows.beat_id.tolist(),key="story_beat")
    row=rows[rows.beat_id==beat_id].iloc[0]
    fig=go.Figure(go.Scatter(x=np.arange(1,351),y=row[FEATURES].to_numpy(dtype=float),name="פעימה אמיתית"))
    fig.add_trace(go.Scatter(x=[126],y=[row.T126],mode="markers+text",text=["R-peak"],textposition="top center",marker=dict(color="#dc6477",size=12),name="R-peak"))
    fig.update_layout(xaxis_title="נקודת זמן T1…T350",yaxis_title="עוצמה מנורמלת")
    chart(fig,"הפעימה שנבחרה מתוך הטבלה, עם סימון פסגת R")
    st.caption(f"הגרף מציג recording_id={recording}, beat_id={beat_id}. זו אותה שורה בטבלה. T126 היא פסגת R; כל שתי עמודות סמוכות מרוחקות 2 מילישניות.")
    st.dataframe(rows[["person_id","recording_id","beat_id"]+FEATURES[:10]],hide_index=True,alt="כל שורה פעימה אמיתית, עם מזהים ועמודות T1 עד T10")
    st.info("לכל פעימה יש 350 ערכים: T1…T350. מוצגות כאן רק T1…T10 לצורך קריאות. האות מסונן וכל פעימה מנורמלת; person_id היא התווית ולא מאפיין קלט.")

    st.header("3. מה מסווגים")
    eligible=model["classes"].tolist()
    left,right=st.columns(2)
    first=left.selectbox("אדם א׳ להשוואה",eligible,key="story_a")
    second=right.selectbox("אדם ב׳ להשוואה",[p for p in eligible if p!=first],key="story_b")
    pair=[first,second]
    fig=go.Figure()
    for person,color in zip(pair,COLORS):
        examples=beats[(beats.person_id==person)&(split=="train")].head(3)
        for j,(_,r) in enumerate(examples.iterrows()):
            fig.add_trace(go.Scatter(x=np.arange(1,351),y=r[FEATURES].to_numpy(dtype=float),
                                    name=person,legendgroup=person,showlegend=j==0,line=dict(color=color),opacity=.65))
    fig.update_layout(xaxis_title="T1…T350",yaxis_title="עוצמה מנורמלת")
    chart(fig,"שלוש פעימות אימון אמיתיות מכל אחד משני אנשים")
    st.markdown("כל פעימה מיוצגת על ידי **350 ערכים**. הרגרסיה הלוגיסטית לומדת מקדם לכל ערך ולכל אדם; כאן מוצגות שתי שורות משקלים מתוך המודל הרב־מחלקתי האמיתי.")
    fig=go.Figure()
    for person,color in zip(pair,COLORS):
        weights=model["coefficients"][eligible.index(person)]
        fig.add_trace(go.Scatter(x=FEATURES,y=weights,name=person,line=dict(color=color)))
    fig.add_hline(y=0,line_dash="dash")
    fig.update_layout(xaxis_title="נקודת זמן T1…T350",yaxis_title="coefficient / weight")
    chart(fig,"המקדמים שנלמדו בפועל לכל אחת מ־350 נקודות הזמן")
    st.caption("אלה מקדמים אחרי StandardScaler: משקל חיובי מעלה את הציון של האדם כשערך המאפיין עולה, ושלילי מוריד אותו. משקל גדול בערך מוחלט מצביע על השפעה גדולה יותר בציון, אך אינו מדד סיבתי לחשיבות: נקודות הזמן תלויות זו בזו. משקל לבדו אינו הסתברות.")

    st.header("4. רגרסיה לוגיסטית – קו שמפריד")
    st.info("PCA משמש להמחשה בלבד. המודל האמיתי עדיין מאומן על כל 350 המאפיינים. ה־StandardScaler וגם ה־PCA הותאמו לנתוני האימון בלבד.")
    fig=go.Figure()
    for person,color in zip(pair,COLORS):
        for group,symbol,label in [("train","circle","אימון"),("test","x","בדיקה")]:
            mask=(beats.person_id==person)&(split==group)
            coordinates=model["coordinates"][mask.to_numpy()]
            fig.add_trace(go.Scatter(x=coordinates[:,0],y=coordinates[:,1],mode="markers",name=f"{person} · {label}",
                                    marker=dict(color=color,symbol=symbol,size=8,opacity=.65),customdata=beats.loc[mask,["recording_id","beat_id"]].to_numpy(),
                                    hovertemplate="%{customdata[0]}<br>beat_id=%{customdata[1]}<extra>%{fullData.name}</extra>"))
    variance=model["explained_variance"]*100
    fig.update_layout(xaxis_title=f"PC1 ({variance[0]:.1f}% מהשונות באימון)",yaxis_title=f"PC2 ({variance[1]:.1f}% מהשונות באימון)")
    chart(fig,"פיזור פעימות של שני אנשים בהטלת PCA לשני ממדים",550)
    st.caption("כל נקודה היא פעימה. לא מוצג קו החלטה: השלכת 350 ממדים לשניים מאבדת מידע, ואין בה בהכרח גבול יחיד שמייצג את החלטת המודל המקורי. לא אימנו מודל חלופי על שני הרכיבים ולא הצגנו אותו כאילו הוא המודל הראשי.")

    st.header("5. עץ החלטה")
    st.info("לא אומן Decision Tree בניסוי הזה. זהו מודל חלופי שמופיע במצגת, ולא נבדק עדיין בפרויקט; אין כאן תוצאות או עץ מאומן להציג.")

    st.header("6. אימון מול בדיקה")
    st.markdown("**הקלטה שלמה שייכת לקבוצה אחת בלבד.** כל הפעימות שלה נשארות יחד, כדי למנוע data leakage.")
    person=st.selectbox("אדם לתרשים חלוקת ההקלטות",eligible,key="story_split_person")
    subset=manifest[(manifest.participant==person)&manifest.split.isin(["train","test"])]
    fig=go.Figure()
    for group,color,label in [("train",COLORS[0],"אימון"),("test",COLORS[1],"בדיקה")]:
        part=subset[subset.split==group]
        fig.add_trace(go.Scatter(x=part.key,y=[label]*len(part),mode="markers",name=label,marker=dict(symbol="square",size=24,color=color)))
    fig.update_layout(xaxis_title="recording_id",yaxis_title="קבוצה")
    chart(fig,"מזהי ההקלטות של האדם לפי אימון ובדיקה",300)
    left,right=st.columns(2)
    for col,group,label in [(left,"train","אימון"),(right,"test","בדיקה")]:
        with col:
            st.subheader(f"{label} — {int((manifest.split==group).sum())} הקלטות")
            st.dataframe(manifest.loc[manifest.split==group,["participant","key","n_beats"]].rename(columns={"participant":"person_id","key":"recording_id","n_beats":"פעימות"}),hide_index=True,alt=f"כל מזהי ההקלטות בקבוצת {label}")
    st.success(f"נבדק: {len(train_ids & test_ids)} הקלטות מופיעות בשתי הקבוצות. לכל אדם בבדיקה קיימת הקלטת אימון.")
    st.caption("החלוקה לפי הקלטות ולא בהכרח לפי ימים. ה־scaler ומשקלי המודל נלמדו רק מהאימון. הקלטות שאינן כשירות לניסוי נשארות מחוץ לשתי הקבוצות.")

    st.header("7. התוצאות שלי")
    metrics=pd.read_csv(folder/"metrics.csv")
    measured=metrics[metrics.unit=="recording"].copy()
    predictions=pd.read_csv(folder/"recording_predictions.csv")
    for _,r in measured.iterrows():
        assert int((predictions[r.method]==predictions.person_id).sum())==r.correct
    st.caption("אותן הקלטות בדיקה בדיוק. Baseline משתמש בתבנית ההקלטה; Logistic Regression ממוצע הסתברויות על פעימות ההקלטה, כפי שחושב בניסוי המקורי.")
    table=measured[["method","correct","total","errors","accuracy","macro_precision","macro_recall","macro_f1-score"]].rename(columns={"method":"שיטה","correct":"נכון","total":"נבדקו","errors":"טעויות","accuracy":"Accuracy","macro_precision":"Precision macro","macro_recall":"Recall macro","macro_f1-score":"F1 macro"})
    st.dataframe(table.style.format({c:"{:.2%}" for c in ["Accuracy","Precision macro","Recall macro","F1 macro"]}),hide_index=True,alt="תוצאות מדודות של שתי השיטות על אותן הקלטות בדיקה")
    for col,method in zip(st.columns(2),["baseline","logistic"]):
        with col:
            st.subheader(method)
            r=measured[measured.method==method].iloc[0]
            st.metric("זיהויים נכונים",f"{int(r.correct)}/{int(r.total)} = {r.accuracy:.2%}")
            cm=pd.read_csv(folder/f"recording_{method}_confusion.csv",index_col=0)
            fig=go.Figure(go.Heatmap(z=cm.to_numpy(),x=cm.columns.tolist(),y=cm.index.tolist(),colorscale="Blues",hovertemplate="אמיתי: %{y}<br>חזוי: %{x}<br>מספר הקלטות: %{z}<extra></extra>"))
            fig.update_layout(xaxis_title="אדם חזוי",yaxis_title="אדם אמיתי",yaxis_autorange="reversed")
            chart(fig,f"מטריצת בלבול מלאה ברמת הקלטה — {method}",520)
            errors=pd.read_csv(folder/f"recording_{method}_errors.csv")
            st.dataframe(errors.rename(columns={"true":"אדם אמיתי","predicted":"זוהה בתור","count":"כמה פעמים"}),hide_index=True,alt=f"כל זוגות השיוך השגוי עבור {method}")
    st.subheader("דוגמאות אמיתיות — True person → Predicted person")
    correct=predictions[predictions.person_id==predictions.logistic].iloc[0]
    wrong=predictions[predictions.person_id!=predictions.logistic]
    st.success(f"זיהוי נכון ברגרסיה הלוגיסטית: {correct.person_id} → {correct.logistic} · הקלטה {correct.recording_id}")
    if not wrong.empty:
        example=wrong.iloc[0]
        st.error(f"טעות ברגרסיה הלוגיסטית: {example.person_id} → {example.logistic} · הקלטה {example.recording_id}")
    else:
        st.info("בהרצה זו אין טעות להציג.")
    st.caption("אלה תוצאות על חלוקה קבועה אחת של אנשים מוכרים; אין להסיק מהן הבטחת זיהוי של אדם זר, ביום אחר או תחת לחץ.")
