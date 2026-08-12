import streamlit as st
import segyio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Geophysical Seismic Data Processing Tool",
    layout="wide"
)

st.title("🗺️ Geophysical Seismic Data Processing & 3D Slicing Tool")
st.markdown("أداة تفاعلية متكاملة لمعالجة وتصوير البيانات الزلزلية ثنائية وثلاثية الأبعاد.")

st.sidebar.header("1. تحميل البيانات")
uploaded_file = st.sidebar.file_uploader("قم برفع ملف SEG-Y (.sgy) هنا", type=["sgy", "segy"])

if uploaded_file is not None:
    try:
        with open("temp_seismic.sgy", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with segyio.open("temp_seismic.sgy", mode='r+', ignore_geometry=True) as segy:
            
            st.subheader("📋 معلومات الملف الزلزالي (File Header Information)")
            
            all_traces = segyio.tools.collect(segy.trace[:])
            n_traces = segy.tracecount
            sample_rate = segyio.tools.dt(segy) / 1000  # ms
            nsamples = segy.samples.size

            col1, col2, col3 = st.columns(3)
            col1.metric("عدد المسارات (N Traces)", f"{n_traces:,}")
            col2.metric("عدد العينات لكل مسار (Samples/Trace)", f"{nsamples:,}")
            col3.metric("معدل أخذ العينات (Sample Rate)", f"{sample_rate:.1f} ms")

            st.sidebar.header("2. سير العمليات (Processing Flow)")
            
            processing_steps = []
            apply_gain = st.sidebar.checkbox("تطبيق كسب تلقائي (AGC)")
            if apply_gain:
                processing_steps.append("AGC")

            apply_norm = st.sidebar.checkbox("تطبيع البيانات (Normalize)")
            if apply_norm:
                processing_steps.append("Normalization")

            # --- 3. إعدادات العرض ---
            st.sidebar.header("3. إعدادات العرض (Visualization Settings)")
            display_traces = st.sidebar.slider(
                "اختر عدد المسارات للعرض:",
                min_value=10,
                max_value=min(n_traces, 1000),
                value=min(n_traces, 200)
            )

            proc_data = all_traces[:display_traces, :].copy()

            if apply_gain:
                rms = np.sqrt(np.mean(proc_data**2, axis=1))
                rms_safe = np.where(rms == 0, 1e-10, rms)
                proc_data = proc_data * (1.0 / rms_safe)[:, np.newaxis]

            if apply_norm:
                max_val = np.max(np.abs(proc_data))
                if max_val > 0:
                    proc_data = proc_data / max_val

            tab1, tab2 = st.tabs(["📊 العرض ثنائي الأبعاد (2D)", "✂️ العرض ثلاثي الأبعاد (3D Slicing)"])

            # ==========================================
            # TAB 1: 2D Section
            # ==========================================
            with tab1:
                st.subheader("📊 المقطع الزلزالي بعد المعالجة (2D)")
                fig, ax = plt.subplots(figsize=(10, 5))
                im = ax.imshow(
                    proc_data.T,
                    cmap='Greys',
                    aspect='auto',
                    extent=[0, display_traces, nsamples * sample_rate, 0]
                )
                fig.colorbar(im, label='Amplitude')
                ax.set_xlabel("Trace Number")
                ax.set_ylabel("Time (ms)")
                ax.grid(True, color='gray', linestyle='--')
                st.pyplot(fig)

                df_export = pd.DataFrame(proc_data).T
                st.download_button(
                    label="📥 تحميل البيانات المعالجة كـ CSV",
                    data=df_export.to_csv(index=False).encode('utf-8'),
                    file_name="processed_seismic_data.csv",
                    mime="text/csv"
                )

            # ==========================================
            # TAB 2: 3D Slicing
            # ==========================================
            with tab2:
                st.subheader("✂️ الشرائح المقطعية التفاعلية (3D)")

                ny = int(np.sqrt(display_traces))
                nx = display_traces // ny

                if nx > 0 and ny > 0:
                    data_3d = proc_data[:nx * ny, :].reshape((nx, ny, nsamples))

                    # أشرطة تحكم داخل التبويب مباشرة لضمان الاستجابة
                    col_x, col_y, col_z = st.columns(3)
                    with col_x:
                        inline_idx = st.slider("Inline (X):", 0, nx - 1, nx // 2)
                    with col_y:
                        crossline_idx = st.slider("Crossline (Y):", 0, ny - 1, ny // 2)
                    with col_z:
                        depth_idx = st.slider("Depth/Time (Z):", 0, nsamples - 1, nsamples // 2)

                    fig_3d = go.Figure()

                    # 1. Inline
                    y_grid, z_grid = np.meshgrid(np.arange(ny), np.arange(nsamples))
                    fig_3d.add_trace(go.Surface(
                        x=np.full_like(y_grid, inline_idx),
                        y=y_grid,
                        z=z_grid,
                        surfacecolor=data_3d[inline_idx, :, :].T,
                        colorscale='RdBu',
                        showscale=False
                    ))

                    # 2. Crossline
                    x_grid, z_grid_y = np.meshgrid(np.arange(nx), np.arange(nsamples))
                    fig_3d.add_trace(go.Surface(
                        x=x_grid,
                        y=np.full_like(x_grid, crossline_idx),
                        z=z_grid_y,
                        surfacecolor=data_3d[:, crossline_idx, :].T,
                        colorscale='RdBu',
                        showscale=False
                    ))

                    # 3. Depth
                    x_grid_z, y_grid_z = np.meshgrid(np.arange(nx), np.arange(ny))
                    fig_3d.add_trace(go.Surface(
                        x=x_grid_z,
                        y=y_grid_z,
                        z=np.full_like(x_grid_z, depth_idx),
                        surfacecolor=data_3d[:, :, depth_idx].T,
                        colorscale='RdBu',
                        colorbar=dict(title="Amplitude")
                    ))

                    fig_3d.update_layout(
                        scene=dict(
                            xaxis=dict(title='Inline / X', range=[0, nx]),
                            yaxis=dict(title='Crossline / Y', range=[0, ny]),
                            zaxis=dict(title='Depth (Samples)', range=[0, nsamples], autorange='reversed'),
                            aspectmode='data'
                        ),
                        margin=dict(l=0, r=0, b=0, t=10),
                        height=650
                    )

                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.warning("يرجى زيادة عدد المسارات للعرض لتشغيل النموذج ثلاثي الأبعاد.")

    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الملف: {e}")
else:
    st.info("👈 يُرجى رفع ملف SEG-Y لبدء المعالجة.")
