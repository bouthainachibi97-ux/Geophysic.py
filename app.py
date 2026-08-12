import streamlit as st
import segyio
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="3D Seismic Data Processing Tool",
    layout="wide"
)

st.title("🗺️ 3D Geophysical Seismic Data Visualization")
st.markdown("أداة تفاعلية لمعالجة وتصوير البيانات الزلزلية في الفضاء ثلاثي الأبعاد (3D Volume).")

st.sidebar.header("1. تحميل البيانات")
uploaded_file = st.sidebar.file_uploader("قم برفع ملف SEG-Y (.sgy) هنا", type=["sgy", "segy"])

if uploaded_file is not None:
    try:
        with open("temp_seismic.sgy", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with segyio.open("temp_seismic.sgy", mode='r+', ignore_geometry=True) as segy:
            st.subheader("📋 معلومات الملف الزلزالي")
            
            all_traces = segyio.tools.collect(segy.trace[:])
            n_traces = segy.tracecount
            sample_rate = segyio.tools.dt(segy) / 1000
            nsamples = segy.samples.size

            col1, col2, col3 = st.columns(3)
            col1.metric("عدد المسارات (N Traces)", f"{n_traces:,}")
            col2.metric("عدد العينات (Samples/Trace)", f"{nsamples:,}")
            col3.metric("معدل أخذ العينات", f"{sample_rate:.1f} ms")

            # --- إعدادات المعالجة ---
            st.sidebar.header("2. معالجة البيانات")
            apply_gain = st.sidebar.checkbox("تطبيق كسب تلقائي (AGC)")
            
            display_traces = st.sidebar.slider(
                "اختر عدد المسارات للعرض 3D:",
                min_value=10,
                max_value=min(n_traces, 500), # حد أقصى للحفاظ على سلاسة الأداء
                value=min(n_traces, 100)
            )

            proc_data = all_traces[:display_traces, :].copy()

            if apply_gain:
                rms = np.sqrt(np.mean(proc_data**2, axis=1))
                rms_safe = np.where(rms == 0, 1e-10, rms)
                proc_data = proc_data * (1.0 / rms_safe)[:, np.newaxis]

            # --- تحويل البيانات إلى شبكة ثلاثية الأبعاد (3D Grid) ---
            # نقوم بإعادة تشكيل المصفوفة لتصبح (X, Y, Z)
            # افترضنا هنا شبكة افتراضية (Grid) بناءً على عدد المسارات والعينات
            grid_side = int(np.sqrt(display_traces))
            if grid_side * grid_side == display_traces:
                data_3d = proc_data.reshape((grid_side, grid_side, nsamples))
            else:
                # إذا لم تكن مربعاً كاملاً، نُنشئ أبعاداً افتراضية للمكعب
                ny = 10 
                nx = display_traces // ny
                data_3d = proc_data[:nx*ny, :].reshape((nx, ny, nsamples))

            # --- عرض المقطع ثلاثي الأبعاد (3D Volume Display) ---
            st.subheader("📊 العرض التفاعلي ثلاثي الأبعاد (Interactive 3D Seismic Cube)")

            X, Y, Z = np.mgrid[0:data_3d.shape[0], 0:data_3d.shape[1], 0:data_3d.shape[2]]

            fig = go.Figure(data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=data_3d.flatten(),
                isomin=np.percentile(data_3d, 10),
                isomax=np.percentile(data_3d, 90),
                opacity=0.1, # شفافية لإظهار باطن المكعب
                surface_count=15, # عدد الطبقات الزلزالية المصورة
                colorscale='Seismic',
                colorbar=dict(title="Amplitude")
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='In-line / X',
                    yaxis_title='Cross-line / Y',
                    zaxis_title='Time / Depth (Samples)',
                    zaxis=dict(autorange='reversed') # عكس محور العمق ليتجه لأسفل
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=700
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"حدث خطأ أثناء المعالجة: {e}")
else:
    st.info("👈 يُرجى رفع ملف SEG-Y لبدء العرض ثلاثي الأبعاد.")
