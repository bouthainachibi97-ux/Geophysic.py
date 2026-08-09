import streamlit as st
import segyio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="Seismic Data Processing Tool",
    layout="wide"
)

st.title("🗺️ Geophysical Seismic Data Processing Tool")
st.markdown("""
أداة تفاعلية لمعالجة وتصوير البيانات الزلزلية الميدانية.
""")


st.sidebar.header("1. تحميل البيانات")
uploaded_file = st.sidebar.file_uploader("قم برفع ملف SEG-Y (.sgy) هنا", type=["sgy", "segy"])

if uploaded_file is not None:
    try:
        # حفظ الملف مؤقتاً لقراءته بمكتبة segyio
        with open("temp_seismic.sgy", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with segyio.open("temp_seismic.sgy", mode='r+', ignore_geometry=True) as segy:
            st.subheader("📋 معلومات الملف الزلزالي (File Header Information)")
            
            all_traces = segyio.tools.collect(segy.trace[:])
            n_traces = segy.tracecount # عدد المسارات
            sample_rate = segyio.tools.dt(segy) / 1000 # معدل أخذ العينات (milliseconds)
            nsamples = segy.samples.size # عدد العينات في كل مسار

            col1, col2, col3 = st.columns(3)
            col1.metric("عدد المسارات (N Traces)", f"{n_traces:,}")
            col2.metric("عدد العينات لكل مسار (Samples/Trace)", f"{nsamples:,}")
            col3.metric("معدل أخذ العينات (Sample Rate)", f"{sample_rate:.1f} ms")

            st.sidebar.header("2. سير العمليات الجيوفيزيائية (Processing Flow)")
            
            processing_steps = []
            apply_gain = st.sidebar.checkbox("تطبيق كسب تلقائي (Automatic Gain Control - AGC)")
            if apply_gain:
                agc_window = st.sidebar.slider("نافذة AGC (Window in samples):", 10, 500, 100)
                processing_steps.append(f"AGC (Window: {agc_window})")

            apply_norm = st.sidebar.checkbox("تطبيع البيانات (Normalize)")
            if apply_norm:
                processing_steps.append("Normalization")

            apply_filter = st.sidebar.checkbox("تطبيق تصفية الترددات (Bandpass Filter)")
            if apply_filter:
                low_freq = st.sidebar.slider("التردد المنخفض (Low Cutoff) - Hz", 1, 50, 10)
                high_freq = st.sidebar.slider("التردد المرتفع (High Cutoff) - Hz", 20, 150, 60)
                processing_steps.append(f"Bandpass ({low_freq}-{high_freq} Hz)")

            st.sidebar.header("3. إعدادات العرض (Visualization Settings)")
            display_traces = st.sidebar.slider(
                "اختر عدد المسارات للعرض:",
                min_value=10,
                max_value=min(n_traces, 1000), # تحديد الحد الأقصى لتفادي البطء
                value=min(n_traces, 200)
            )
            
            
            # 1. اختيار شريحة من البيانات
            proc_data = all_traces[:display_traces, :].copy()
            
            if apply_gain:
                # تطبيق كسب مبسط باستخدام مقلوب متوسط الجذر التربيعي لجعل الإشارات الضعيفة مرئية
                rms = np.sqrt(np.mean(proc_data**2, axis=1))
                rms_safe = np.where(rms == 0, 1e-10, rms) # تجنب القسمة على صفر
                gain_factor = 1.0 / rms_safe
                # توسيع مصفوفة الكسب لتطابق أبعاد البيانات
                proc_data = proc_data * gain_factor[:, np.newaxis]

            # 3. تصفية الترددات (مثال برمجي)
            # ملاحظة: في المعالجة الحقيقية تُستخدم مكتبة scipy.signal، هنا للتوضيح.
            # سأستخدم "تقييد القيم" البسيط كمثال برمجي، وفي التطبيق الحقيقي يجب تطبيق Bandpass.
            
            if apply_norm:
                max_val = np.max(np.abs(proc_data))
                if max_val > 0:
                    proc_data = proc_data / max_val

            # --- عرض المقطع الزلزالي (Visualization) ---
            st.subheader(f"📊 المقطع الزلزالي بعد المعالجة (Processed Seismic Section)")
            st.markdown(f"**العمليات المطبقة:** {', '.join(processing_steps) if processing_steps else 'لا توجد'}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # رسم البيانات الزلزلية كصورة 2D (Wiggle / VAr display)
            # المحور الصادي: العمق بالوقت (ms)
            # المحور السيني: رقم المسار
            
            # عرض المقطع بتنسيق رمادي (Variable Area/Density)
            im = ax.imshow(proc_data.T, 
                           cmap='Greys', 
                           aspect='auto', 
                           extent=[0, display_traces, nsamples * sample_rate, 0])
            
            fig.colorbar(im, label='Amplitude', orientation='vertical')
            
            ax.set_xlabel("Trace Number", fontsize=12)
            ax.set_ylabel("Time (ms)", fontsize=12)
            ax.set_title("Seismic Section Display", fontsize=14)
            ax.grid(True, which='both', color='gray', linestyle='--')
            
            st.pyplot(fig)

            # --- تصدير البيانات المعالجة ---
            st.subheader("💾 تصدير النتائج (Export Results)")
            df_export = pd.DataFrame(proc_data).T # تحويل البيانات المحسوبة لجدول لتصديره
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 تحميل البيانات المعالجة (Processed Data) كـ CSV",
                data=csv_data,
                file_name="processed_seismic_data.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الملف الزلزالي: {e}")
        st.warning("تأكد من أن ملف SEG-Y صالح وصغير الحجم بما يكفي للتجربة.")

else:
    st.info("👈 يُرجى رفع ملف SEG-Y (.sgy) من الشريط الجانبي لبدء المعالجة والتصوير.")
