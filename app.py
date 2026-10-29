from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ecg_baseline import analyze

st.set_page_config(page_title="HeartPrint | ECG",page_icon="💓",layout="wide")
st.markdown("""<style>
.stApp {background: #f5f7fb; color:#18304b;}
h1,h2,h3,p,[data-testid="stCaptionContainer"],label {direction:rtl; text-align:right;}
.block-container {max-width:1500px;padding-top:2rem;}
[data-testid="stMetric"] {background:white;border:1px solid #e1e7ef;border-top:4px solid #209d91;border-radius:14px;padding:18px;}
[data-testid="stMetricLabel"] {font-size:1rem;}
[data-baseweb="tab-list"] {gap:12px;flex-wrap:wrap;}
[data-baseweb="tab"] {background:white;border-radius:10px;padding:14px 20px;}
.hero {direction:rtl;background:linear-gradient(120deg,#142c4b,#166e77);color:white;padding:28px 34px;border-radius:22px;margin:0 0 24px;}
.hero h1 {color:white;font-size:2.1rem;margin:0;}
.hero p {color:#d7edef;margin:10px 0 0;font-size:1.1rem;}
.verdict {direction:rtl;background:#e6f5ef;border-right:5px solid #209d91;border-radius:12px;padding:18px 22px;margin:18px 0; font-size:1.15rem;}
</style>""",unsafe_allow_html=True)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ecg-id-database-1.0.0"

# Streamlit can rerun app.py while an older helper module remains in sys.modules.
# Reload only when that module's source differs from the version it loaded.
import session3_dashboard as session3
dashboard_source = Path(session3.__file__)
dashboard_signature = (dashboard_source.stat().st_mtime_ns, dashboard_source.stat().st_size)
if getattr(session3, "SOURCE_SIGNATURE", None) != dashboard_signature:
    session3 = importlib.reload(session3)

view = st.sidebar.radio("תצוגה", ["לפי מצגת מפגש 3", "פירוק הקלטות לטבלת פעימות", "מפגש 3 — השוואת מודלים", "הניתוח המקורי"])
if view == "לפי מצגת מפגש 3":
    session3.render_slide_walkthrough(ROOT)
    st.stop()
if view == "פירוק הקלטות לטבלת פעימות":
    session3.render_beat_table(ROOT)
    st.stop()
if view == "מפגש 3 — השוואת מודלים":
    session3.render_comparison(ROOT)
    st.stop()

@st.cache_data(show_spinner=False)
def load_analysis(signature):
    return analyze(DATA)

if not DATA.is_dir():
    st.error("תיקיית מאגר ECG-ID לא נמצאה ליד האפליקציה."); st.stop()
# Cache invalidates when source file sizes or modification times change.
signature = tuple((str(p.relative_to(DATA)),p.stat().st_size,p.stat().st_mtime_ns)
                  for pattern in ("Person_*/*.hea","Person_*/*.dat") for p in sorted(DATA.glob(pattern)))
st.markdown('<div class="hero"><h1>💓 HeartPrint | לכל לב יש צורה משלו?</h1><p>מהקלטת ECG לפעימה אישית — בודקים מי דומה לעצמו, מי דומה לאחרים ועד כמה הזיהוי מצליח.</p></div>',unsafe_allow_html=True)
with st.spinner("קורא את ההקלטות, מחלץ פעימות ומחשב את ניסוי הזיהוי…"):
    try:
        data = load_analysis(signature)
    except Exception as exc:
        st.error(f"לא ניתן להשלים את הניתוח: {exc}"); st.stop()
records, results = data["recordings"], data["results"]
accuracy = results.correct.mean()
a,b,c,d = st.columns(4)
a.metric("משתתפים בניסוי",len(data["labels"]))
b.metric("הקלטות בדיקה",len(results))
c.metric("דיוק בזיהוי",f"{accuracy:.2%}")
d.metric("זיהויים נכונים",f"{results.correct.sum()} / {len(results)}")
st.markdown(f'<div class="verdict"><strong>תוצאת הניסוי: {int(results.correct.sum())} מתוך {len(results)} הקלטות זוהו נכון.</strong><br>השוואת צורת הפעימה מאפשרת זיהוי ברוב המקרים בניסוי הזה. מדובר באנשים שנרשמו מראש, ובהקלטות בדיקה נפרדות.</div>',unsafe_allow_html=True)
with st.expander("איך לקרוא את התוצאות?"):
    st.write("כל הקלטה נמצאת רק באימון או בבדיקה. לרוב המשתתפים שתי הקלטות בלבד, ולכן החלוקה שלהם היא 50%/50%. ירוק = זיהוי נכון; אדום = זיהוי שגוי. דמיון אינו הסתברות: 0.99 אינו אומר 99% ביטחון.")

def chart(fig, height=430):
    fig.update_layout(template="plotly_white",height=height,margin=dict(l=40,r=35,t=65,b=65),font=dict(family="Arial",size=14,color="#18304b"),title=dict(x=.98,xanchor="right"),colorway=["#2687cf","#e99529","#23a18d","#9169c1","#dc6477"],legend=dict(orientation="h",y=-.25),paper_bgcolor="white",plot_bgcolor="white",hoverlabel=dict(font_size=14))
    st.plotly_chart(fig,width="stretch",config={"displaylogo":False,"scrollZoom":False})

def template_plot(keys, title):
    fig = go.Figure()
    for key in keys:
        item = data["beats"][key]
        fig.add_trace(go.Scatter(x=item["time"],y=item["vector"],name=key))
    fig.update_layout(title=title,xaxis_title="זמן ביחס לשיא R (שניות)",yaxis_title="תבנית מנורמלת")
    chart(fig)

tabs = st.tabs(["① כך נראית פעימה","② אדם מול עצמו","③ אדם מול אחרים","④ האם הזיהוי הצליח?","⑤ הנתונים והשיטה"])
with tabs[0]:
    st.subheader("מחזור לב מתוך הקלטה אמיתית")
    st.caption("הנקודות מסמנות פסגות שזוהו אוטומטית. האזור הכתום הוא חלון של פעימה אחת; מתחתיו רואים את הפעימות כשהן מיושרות זו לזו.")
    key = st.selectbox("בחרי הקלטה",list(data["beats"]),key="beat_record")
    item = data["beats"][key]
    window = st.slider("קטע ההקלטה לתצוגה (שניות)",0.,float(len(item["clean"])/item["fs"]), (0.,5.),step=.5)
    t = np.arange(len(item["clean"]))/item["fs"]
    mask = (t>=window[0]) & (t<=window[1])
    peaks = item["peaks"]; ptime = peaks/item["fs"]
    shown = peaks[(ptime>=window[0]) & (ptime<=window[1])]
    fig = go.Figure(go.Scatter(x=t[mask],y=item["clean"][mask],name="ECG מסונן"))
    fig.add_trace(go.Scatter(x=shown/item["fs"],y=item["clean"][shown],mode="markers",name="פסגות אוטומטיות",marker=dict(color="#e65b65",size=9)))
    if len(shown):
        center = shown[0]/item["fs"]
        fig.add_vrect(x0=center-.25,x1=center+.45,fillcolor="orange",opacity=.15,line_width=0)
    fig.update_layout(xaxis_title="זמן (שניות)",yaxis_title="משרעת ביחידות האות",title="איתור שיאים וחלון פעימה")
    chart(fig)
    fig = go.Figure()
    for beat in item["beats"][:20]:
        fig.add_trace(go.Scatter(x=item["time"],y=beat,line=dict(color="rgba(40,120,160,.2)"),showlegend=False))
    fig.add_trace(go.Scatter(x=item["time"],y=item["mean_beat"],name="פעימה ממוצעת",line=dict(color="#12283f",width=3)))
    fig.update_layout(xaxis_title="זמן ביחס לשיא R (שניות)",yaxis_title="משרעת מנורמלת",title=f"פעימות מיושרות — {len(item['beats'])} פעימות שימושיות")
    chart(fig)
    st.caption("הפסגות מזוהות אוטומטית, ולא אומתו ידנית. החלון הוא 250 מילישניות לפני השיא ו־450 אחריו.")
with tabs[1]:
    st.subheader("האם אותה צורה חוזרת בהקלטה אחרת?")
    st.caption("בחרי אדם ושתי הקלטות. חפיפה בין הקווים מעידה על צורת פעימה דומה גם כשההקלטה השתנתה.")
    person = st.selectbox("משתתף להשוואה בין הקלטות",data["labels"])
    keys = records.loc[(records.participant==person)&(records.error==""),"key"].tolist()
    chosen = st.multiselect("הקלטות להשוואה",keys,default=keys[:2])
    if chosen:
        template_plot(chosen,"תבניות של אותו אדם בהקלטות שונות")
    if len(chosen)>=2:
        vectors = np.vstack([data["beats"][k]["vector"] for k in chosen])
        st.metric("דמיון בין שתי ההקלטות הראשונות",f"{vectors[0] @ vectors[1]:.4f}")
        with st.expander("טבלת דמיון לכל ההקלטות שנבחרו"):
            st.dataframe(pd.DataFrame(vectors@vectors.T,index=chosen,columns=chosen).round(4),width="stretch")
    st.caption("דמיון קרוב ל־1 מציין צורה דומה לאחר נרמול. להשוואה זו מותר לבחור כל הקלטה; תוצאות ניסוי הזיהוי נשארות קבועות ונפרדות.")
with tabs[2]:
    st.subheader("מה מבדיל בין חתימות של אנשים שונים?")
    st.caption("כל צבע מייצג משתתף. חפשי הבדלים בגובה היחסי, ברוחב ובצורה של הגלים סביב השיא. לחיצה על שם במקרא מסתירה או מציגה קו.")
    chosen = st.multiselect("משתתפים להשוואה",data["labels"],default=data["labels"][:5])
    if chosen:
        fig = go.Figure()
        time = data["beats"][data["train"][0]]["time"]
        for person in chosen:
            fig.add_trace(go.Scatter(x=time,y=data["gallery"][person],name=person))
        fig.update_layout(title="תבניות משתתפים מהאימון בלבד",xaxis_title="זמן ביחס לשיא R (שניות)",yaxis_title="תבנית מנורמלת")
        chart(fig)
        vectors = np.vstack([data["gallery"][p] for p in chosen])
        similarity = np.clip(vectors@vectors.T,-1,1)
        fig = go.Figure(go.Heatmap(z=similarity,x=chosen,y=chosen,zmin=-1,zmax=1,colorscale="RdBu",reversescale=True,colorbar=dict(title="דמיון"),text=np.round(similarity,3),texttemplate="%{text}" if len(chosen)<=10 else None,hovertemplate="%{x}<br>%{y}<br>דמיון: %{z:.4f}<extra></extra>"))
        fig.update_layout(title="דמיון בין תבניות האימון",yaxis=dict(autorange="reversed"))
        chart(fig,height=max(430,min(900,30*len(chosen))))
    st.markdown("### השוואה פשוטה: שני אנשים, שני קווים")
    left,right = st.columns(2)
    pa=left.selectbox("משתתף ראשון",data["labels"],index=0)
    pb=right.selectbox("משתתף שני",data["labels"],index=1)
    fig=go.Figure()
    for person in dict.fromkeys([pa,pb]):
        fig.add_trace(go.Scatter(x=data["beats"][data["train"][0]]["time"],y=data["gallery"][person],name=person))
    fig.update_layout(title=f"דמיון בין התבניות: {data['gallery'][pa] @ data['gallery'][pb]:.4f}",xaxis_title="זמן ביחס לשיא R (שניות)",yaxis_title="תבנית מנורמלת")
    chart(fig)
    st.caption("האלכסון במפת האימון הוא 1 בהגדרה. הוא אינו הוכחה ליציבות בין הקלטות.")
with tabs[3]:
    st.subheader("ניסוי זיהוי בהקלטות שלא שימשו לבניית התבניות")
    fig = go.Figure(go.Pie(labels=["נכון","שגוי"],values=[int(results.correct.sum()),int((~results.correct).sum())],hole=.72,marker=dict(colors=["#209d91","#dc6477"]),textinfo="label+value",sort=False))
    fig.update_layout(title="כמה הקלטות זוהו נכון?",annotations=[dict(text=f"<b>{accuracy:.1%}</b><br>דיוק בזיהוי",x=.5,y=.5,font_size=24,showarrow=False)])
    chart(fig)
    cm = data["confusion"]
    st.markdown("### איפה המערכת מתבלבלת?")
    st.caption("במטריצה, שורה היא האדם האמיתי ועמודה היא האדם שנבחר. תאים על האלכסון הם הצלחות; תאים מחוץ לו הם טעויות. כברירת מחדל מוצגים רק האנשים המעורבים בטעויות.")
    if not st.checkbox("הצגת כל המשתתפים במטריצה",value=False):
        errors=results.loc[~results.correct]
        involved=sorted(set(errors.true)|set(errors.predicted))
        if involved:
            cm=cm.loc[involved,involved]
    fig=go.Figure(go.Heatmap(z=cm.to_numpy(),x=cm.columns,y=cm.index,colorscale="Blues",hovertemplate="אמיתי: %{y}<br>חזוי: %{x}<br>הקלטות: %{z}<extra></extra>"))
    fig.update_layout(title="מטריצת בלבול — אפשר להגדיל אזור ולרחף על תא",xaxis_title="משתתף חזוי",yaxis_title="משתתף אמיתי",yaxis=dict(autorange="reversed"))
    chart(fig,620)
    mode=st.radio("הקלטות להצגה",["כולן","נכון","שגוי"],horizontal=True)
    shown=results if mode=="כולן" else results[results.correct == (mode=="נכון")]
    st.dataframe(shown,width="stretch",hide_index=True)
    if not shown.empty:
        key=st.selectbox("בדיקת זיהוי בודד",shown.recording.tolist())
        row=results.loc[results.recording==key].iloc[0]
        if row["correct"]:
            st.success(f"✓ זיהוי נכון — {row['true']}")
        else:
            st.error(f"✕ זיהוי שגוי — האדם הוא {row['true']}, אך נבחר {row['predicted']}")
        st.caption("כחול: ההקלטה החדשה · ירוק מקווקו: תבנית האדם האמיתי · אדום: התבנית שנבחרה בטעות, אם קיימת.")
        fig=go.Figure(go.Scatter(x=data["beats"][key]["time"],y=data["beats"][key]["vector"],name="הקלטה חדשה",line=dict(color="#2687cf",width=3)))
        for p in dict.fromkeys([row["true"],row["predicted"]]):
            correct=p==row["true"]
            fig.add_trace(go.Scatter(x=data["beats"][key]["time"],y=data["gallery"][p],name=("האדם האמיתי: " if correct else "זיהוי שגוי: ")+p,line=dict(color="#209d91" if correct else "#dc6477",width=2.5,dash="dash" if correct else "solid")))
        fig.update_layout(title=f"אמיתי: {row['true']} | חזוי: {row['predicted']}",xaxis_title="זמן ביחס לשיא R (שניות)",yaxis_title="תבנית מנורמלת")
        chart(fig)
    st.write(f"דמיון עצמי ממוצע: {results.self_similarity.mean():.3f}; דמיון ממוצע לאדם האחר הקרוב ביותר: {results.best_other_similarity.mean():.3f}.")
    st.warning("רוב ההקלטות נאספו באותו יום. התוצאה אינה מוכיחה זיהוי לאורך זמן, תחת מאמץ או של אדם שלא הופיע באימון. תצוגת הבחירות אינה משנה את הניסוי.")
with tabs[4]:
    st.write(f"במאגר {records.participant.nunique()} משתתפים ו־{len(records)} הקלטות. בניסוי {len(data['train'])} הקלטות אימון ו־{len(data['test'])} הקלטות בדיקה.")
    st.dataframe(records.groupby("participant").agg(recordings=("key","size"),usable=("error",lambda s:(s=="").sum()),age=("age","first"),sex=("sex","first")),width="stretch")
    st.write("תדרי דגימה שנמצאו:",sorted(records.fs.unique().tolist()))
    st.write("אורכי הקלטה בשניות:",sorted(records.duration.unique().tolist()))
    st.write("ערוצים: ECG I ו־ECG I filtered. הניסוי משתמש בערוץ המסונן בלבד; אלה שתי גרסאות של אותה מדידה.")
    failed=records[records.error!=""]
    st.write(f"הקלטות שלא עברו חילוץ: {len(failed)}")
    if not failed.empty: st.dataframe(failed[["key","error"]],width="stretch")
    st.markdown("**השיטה:** סינון אחיד, איתור פסגות QRS, יישור חלונות, נרמול פעימות, ממוצע לכל הקלטה וממוצע שווה־משקל בין הקלטות האימון של כל אדם. הניבוי הוא התבנית עם דמיון הקוסינוס הגבוה ביותר.")
    st.caption("האפליקציה מקומית ואינה מעלה את האותות לשירות חיצוני. אין העלאת קבצים, אימון מודל או שינוי של קובצי המקור.")

