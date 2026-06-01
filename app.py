import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import os
import trafilatura

# إعدادات الصفحة
st.set_page_config(page_title="كاشف الأخبار بالروابط", page_icon="🌐", layout="centered")

# 1. تحميل موديل الـ Embedding
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

model = load_embedding_model()

# 2. دالة استخراج النص من أي رابط (Web Scraping)
def extract_text_from_url(url):
    try:
        # تحميل محتوى الرابط
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None
        # استخلاص النص الأساسي للمقال فقط (بدون إعلانات أو قوائم جانبية)
        result = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        return result
    except Exception as e:
        return None

# 3. دالة قراءة وتحديث البيانات تلقائياً من ملف CSV
def load_and_vectorize_data():
    csv_file = "news_dataset.csv"
    if not os.path.exists(csv_file):
        df_default = pd.DataFrame(columns=["text", "status", "source", "is_fake_indicator"])
        df_default.to_csv(csv_file, index=False, encoding='utf-8-sig')
        return df_default, None
    
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    if df.empty:
        return df, None
        
    db_texts = df["text"].tolist()
    db_embeddings = model.encode(db_texts)
    dimension = db_embeddings.shape[1]
    
    vector_space = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(np.array(db_embeddings).astype('float32'))
    vector_space.add(np.array(db_embeddings).astype('float32'))
    
    return df, vector_space

df_news, vector_space = load_and_vectorize_data()

# --- واجهة المستخدم (Streamlit UI) ---
st.title("🌐 نظام فحص الأخبار عبر الروابط والنصوص")
st.write(f"📊 يحتوي فضاء المتجهات الحالي على: **{len(df_news)}** خبر موثق.")

# قائمة جانبية لإضافة البيانات
with st.sidebar:
    st.header("➕ إضافة خبر رسمي جديد")
    new_text = st.text_area("نص البلاغ الرسمي:")
    new_source = st.text_input("المصدر الرسمي:")
    new_type = st.selectbox("نوع الخبر:", ["حقيقة داحضة للشائعة", "حقيقة مؤكدة"])
    
    if st.button("إضافة وتحديث الفضاء 📥"):
        if new_text.strip() == "" or new_source.strip() == "":
            st.error("الرجاء ملء جميع الحقول!")
        else:
            is_fake = True if new_type == "حقيقة داحضة للشائعة" else False
            new_row = pd.DataFrame([{"text": new_text, "status": new_type, "source": new_source, "is_fake_indicator": is_fake}])
            new_row.to_csv("news_dataset.csv", mode='a', header=False, index=False, encoding='utf-8-sig')
            st.success("تمت الإضافة بنجاح!")
            st.rerun()

# تبويب الفحص (نص عادي أو رابط)
input_mode = st.radio("اختر طريقة الفحص:", ["فحص عبر لصق رابط المقال (URL)", "فحص عبر نص مكتوب"])

user_input = ""

if input_mode == "فحص عبر لصق رابط المقال (URL)":
    url_input = st.text_input("أدخل رابط المقال الإخباري المُراد فحصه:", placeholder="https://example.com")
    if url_input:
        with st.spinner("جاري قراءة الرابط واستخراج نص المقال تلقائياً..."):
            extracted_text = extract_text_from_url(url_input)
            if extracted_text:
                st.info("📝 **النص الذي تم استخراجه من الرابط بنجاح:**")
                # عرض أول 300 حرف فقط كمعاينة
                st.write(extracted_text[:300] + "...") 
                user_input = extracted_text
            else:
                st.error("❌ فشل النظام في قراءة محتوى هذا الرابط. تأكد من أن الرابط يعمل وصالح للاستخدام العام.")
else:
    user_input = st.text_area("أدخل نص الخبر المشكوك فيه:")

# زر تفعيل الفحص عبر فضاء المتجهات
if st.button("ابدأ الفحص الذكي 🚀"):
    if df_news.empty or vector_space is None:
        st.error("قاعدة البيانات فارغة حالياً. أضف أخباراً موثوقة أولاً من القائمة الجانبية.")
    elif user_input.strip() == "":
        st.warning("الرجاء إدخال رابط صالح أو نص للفحص!")
    else:
        with st.spinner("جاري تحليل الأبعاد والمقارنة الرياضية..."):
            # تحويل النص (سواء مدخل أو مستخرج من الرابط) إلى متجهة
            query_embedding = model.encode([user_input])
            faiss.normalize_L2(np.array(query_embedding).astype('float32'))

            # البحث عن النتيجة في فضاء المتجهات
            similarities, indices = vector_space.search(np.array(query_embedding).astype('float32'), k=1)
            
            best_match_idx = indices[0][0]
            similarity_score = similarities[0][0]
            matched_news = df_news.iloc[best_match_idx]

            st.subheader("📊 نتيجة المطابقة السيمانتيكية:")
            st.metric(label="نسبة التطابق المعنوي مع قاعدة البيانات", value=f"{similarity_score * 100:.2f}%")

            if similarity_score > 0.60:
                if matched_news["is_fake_indicator"]:
                    st.error("🚨 النتيجة المحتملة: المقال ينشر أخباراً زنيفة / إشاعة تفندها البلاغات الرسمية!")
                    st.info(f"**التوضيح الرسمي المخزن:** {matched_news['text']}")
                else:
                    st.success("✅ النتيجة المحتملة: المقال متوافق مع الأخبار الرسمية المؤكدة.")
                    st.info(f"**المرجع المطابق:** {matched_news['text']}")
                
                st.caption(f"**المصدر المرجعي للتحقق:** {matched_news['source']}")
            else:
                st.warning("⚠️ النتيجة: لا توجد بلاغات رسمية قريبة من سياق هذا المقال في فضاء المتجهات حالياً لحسم صحته.")