import streamlit as st
import segyio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="Geophysical Seismic Data Processing Tool",
    layout="wide"
)

st.title("🗺️ Geophysical Seismic Data Processing & 3D Slicing Tool")
st.markdown("أداة تفاعلية متكاملة لمعالجة وتصوير البيانات الزلزلية ثنائية وثلاثية الأبعاد.")

# --- 1. الشريط الجانبي لتحميل البيانات ---
st.sidebar.header("1. تحميل البيانات")
uploaded_file = st.sidebar.file_uploader("قم برفع ملف SEG-Y (.sgy) هنا", type=["sgy", "segy"])

if uploaded_file is not None:
    try:
        # حفظ الملف مؤقتاً لقراءته بمكتبة segyio
        with open("temp_seismic.sgy", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with segyio.open("temp_seismic.sgy", mode='r+', ignore_geometry=True) as segy:
            
            # --- معلومات الملف الزلزالي ---
            st.subheader("📋 معلومات الملف الزلزالي (File Header Information)")
            
            all_traces = segyio.tools.collect(segy.trace[:])
            n_traces = segy.tracecount
            sample_rate = segyio.tools.dt(segy) / 1000  # milliseconds
            nsamples = segy.samples.size

            col1, col2, col3 = st.columns(3)
            col1.metric("عدد المسارات (N Traces)", f"{n_traces:,}")
            col2.metric("عدد العينات لكل مسار (Samples/Trace)", f"{nsamples:,}")
            col3.metric("معدل أخذ العينات (Sample Rate)", f"{sample_rate:.1f} ms")

            # --- 2. معالجة البيانات الجيوفيزيائية ---
            st.sidebar.header("2. سير العمليات (Processing Flow)")
            
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

            # --- 3. إعدادات العرض ---
            st.sidebar.header("3. إعدادات العرض (Visualization Settings)")
            display_traces = st.sidebar.slider(
                "اختر عدد المسارات للعرض:",
                min_value=10,
                max_value=min(n_traces, 1000),
                value=min(n_traces, 200)
            )

            # معالجة الشريحة المحددة من البيانات
            proc_data = all_traces[:display_traces, :].copy()

            if apply_gain:
                rms = np.sqrt(np.mean(proc_data**2, axis=1))
                rms_safe = np.where(rms == 0, 1e-10, rms)
                gain_factor = 1.0 / rms_safe
                proc_data = proc_data * gain_factor[:, np.newaxis]

            if apply_norm:
                max_val = np.max(np.abs(proc_data))
                if max_val > 0:
                    proc_data = proc_data / max_val

            # --- تقسيم الواجهة إلى علامات تبويب (Tabs) ---
            tab1, tab2 = st.tabs(["📊 العرض ثنائي الأبعاد (2D Section)", "✂️ العرض ثلاثي الأبعاد والشرائح (3D Slicing)"])

            # ==========================================
            # TAB 1: العرض ثنائي الأبعاد 2D
            # ==========================================
            with tab1:
                st.subheader("📊 المقطع الزلزالي بعد المعالجة (2D Seismic Section)")
                st.markdown(f"**العمليات المطبقة:** {', '.join(processing_steps) if processing_steps else 'لا توجد'}")

                fig, ax = plt.subplots(figsize=(10, 6))
                im = ax.imshow(
                    proc_data.T,
                    cmap='Greys',
                    aspect='auto',
                    extent=[0, display_traces, nsamples * sample_rate, 0]
                )
                
                fig.colorbar(im, label='Amplitude', orientation='vertical')
                ax.set_xlabel("Trace Number", fontsize=12)
                ax.set_ylabel("Time (ms)", fontsize=12)
                ax.set_title("Seismic Section Display", fontsize=14)
                ax.grid(True, which='both', color='gray', linestyle='--')

                st.pyplot(fig)

                # تصدير البيانات
                st.subheader("💾 تصدير النتائج (Export Results)")
                df_export = pd.DataFrame(proc_data).T
                csv_data = df_export.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📥 تحميل البيانات المعالجة (Processed Data) كـ CSV",
                    data=csv_data,
                    file_name="processed_seismic_data.csv",
                    mime="text/csv"
                )

            # ==========================================
            # TAB 2: العرض ثلاثي الأبعاد والشرائح 3D
            # ==========================================
            with tab2:
                st.subheader("✂️ الشرائح المقطعية التفاعلية (3D Orthogonal Slicing)")

                # إعادة تشكيل البيانات إلى شبكة 3D (Inline x Crossline x Depth)
                ny = int(np.sqrt(display_traces))
                nx = display_traces // ny
                if nx * ny > 0:
                    data_3d = proc_data[:nx * ny, :].reshape((nx, ny, nsamples))

                    st.sidebar.markdown("---")
                    st.sidebar.header("🎛️ موضع الشرائح 3D")
                    inline_idx = st.sidebar.slider("Inline Slice (X):", 0, nx - 1, nx // 2)
                    crossline_idx = st.sidebar.slider("Crossline Slice (Y):", 0, ny - 1, ny // 2)
                    depth_idx = st.sidebar.slider("Depth/Time Slice (Z):", 0, nsamples - 1, nsamples // 2)

                    SEISMIC_COLORSCALE = 'RdBu'
                    fig_3d = go.Figure()

                    # 1. Inline Slice
                    y_grid, z_grid = np.meshgrid(np.arange(ny), np.arange(nsamples))
                    x_inline = np.full_like(y_grid, inline_idx)
                    inline_colors = data_3d[inline_idx, :, :].T

                    fig_3d.add_trace(go.Surface(
                        x=x_inline, y=y_grid, z=z_grid,
                        surfacecolor=inline_colors,
                        colorscale=SEISMIC_COLORSCALE, showscale=False,
                        name='Inline Slice'
                    ))

                    # 2. Crossline Slice
                    x_grid, z_grid_y = np.meshgrid(np.arange(nx), np.arange(nsamples))
                    y_crossline = np.full_like(x_grid, crossline_idx)
                    crossline_colors = data_3d[:, crossline_idx, :].T

                    fig_3d.add_trace(go.Surface(
                        x=x_grid, y=y_crossline, z=z_grid_y,
                        surfacecolor=crossline_colors,
                        colorscale=SEISMIC_COLORSCALE, showscale=False,
                        name='Crossline Slice'
                    ))

                    # 3. Depth Slice
                    x_grid_z, y_grid_z = np.meshgrid(np.arange(nx), np.arange(ny))
                    z_depth = np.full_like(x_grid_z, depth_idx)
                    depth_colors = data_3d[:, :, depth_idx].T

                    fig_3d.add_trace(go.Surface(
                        x=x_grid_z, y=y_grid_z, z=z_depth,
                        surfacecolor=depth_colors,
                        colorscale=SEISMIC_COLORSCALE,
                        colorbar=dict(title="Amplitude"),
                        name='Depth Slice'
                    ))

                    fig_3d.update_layout(
                        scene=dict(
                            xaxis=dict(title='Inline / X', range=[0, nx]),
                            yaxis=dict(title='Crossline / Y', range=[0, ny]),
                            zaxis=dict(title='Depth / Time (Samples)', range=[0, nsamples], autorange='reversed'),
                            aspectmode='data'
                        ),
                        margin=dict(l=0, r=0, b=0, t=30),
                        height=700
                    )

                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.warning("عدد المسارات المختارة غير كافٍ لتشكيل شبكة ثلاثية الأبعاد. يرجى زيادة عدد المسارات للعرض.")

    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الملف الزلزالي: {e}")
        st.warning("تأكد من أن ملف SEG-Y صالح وصغير الحجم بما يكفي للتجربة.")

else:
    st.info("👈 يُرجى رفع ملف SEG-Y (.sgy) من الشريط الجانبي لبدء المعالجة والتصوير.")
