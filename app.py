import streamlit as st
import segyio
import numpy as np
import plotly.graph_objects as go

# --- إعدادات الصفحة ---
st.set_page_config(page_title="3D Seismic Slicing Tool", layout="wide")
st.title("✂️ 3D Seismic Orthogonal Slicing (Inline, Crossline, Depth)")

uploaded_file = st.sidebar.file_uploader("قم برفع ملف SEG-Y (.sgy)", type=["sgy", "segy"])

if uploaded_file is not None:
    try:
        with open("temp_seismic.sgy", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with segyio.open("temp_seismic.sgy", mode='r+', ignore_geometry=True) as segy:
            all_traces = segyio.tools.collect(segy.trace[:])
            n_traces = segy.tracecount
            nsamples = segy.samples.size

            # تحويل البيانات إلى مكعب 3D مفترض (Inlines x Crosslines x Depth)
            ny = int(np.sqrt(n_traces))
            nx = n_traces // ny
            data_3d = all_traces[:nx*ny, :].reshape((nx, ny, nsamples))

            # --- أشرطة التحكم الجانبية لتحديد موضع الشرائح ---
            st.sidebar.header("🎛️ موضع الشرائح (Slice Controls)")
            
            inline_idx = st.sidebar.slider("Inline Slice (X):", 0, nx - 1, nx // 2)
            crossline_idx = st.sidebar.slider("Crossline Slice (Y):", 0, ny - 1, ny // 2)
            depth_idx = st.sidebar.slider("Depth/Time Slice (Z):", 0, nsamples - 1, nsamples // 2)

            fig = go.Figure()

            # 1. شريحة Inline Slice (مقطع X ثابت)
            y_grid, z_grid = np.meshgrid(np.arange(ny), np.arange(nsamples))
            x_inline = np.full_like(y_grid, inline_idx)
            inline_colors = data_3d[inline_idx, :, :].T

            fig.add_trace(go.Surface(
                x=x_inline, y=y_grid, z=z_grid,
                surfacecolor=inline_colors,
                colorscale='Seismic', showscale=False,
                name='Inline Slice'
            ))

            # 2. شريحة Crossline Slice (مقطع Y ثابت)
            x_grid, z_grid_y = np.meshgrid(np.arange(nx), np.arange(nsamples))
            y_crossline = np.full_like(x_grid, crossline_idx)
            crossline_colors = data_3d[:, crossline_idx, :].T

            fig.add_trace(go.Surface(
                x=x_grid, y=y_crossline, z=z_grid_y,
                surfacecolor=crossline_colors,
                colorscale='Seismic', showscale=False,
                name='Crossline Slice'
            ))

            # 3. شريحة Depth Slice (مقطع Z أفقي)
            x_grid_z, y_grid_z = np.meshgrid(np.arange(nx), np.arange(ny))
            z_depth = np.full_like(x_grid_z, depth_idx)
            depth_colors = data_3d[:, :, depth_idx].T

            fig.add_trace(go.Surface(
                x=x_grid_z, y=y_grid_z, z=z_depth,
                surfacecolor=depth_colors,
                colorscale='Seismic',
                colorbar=dict(title="Amplitude"),
                name='Depth Slice'
            ))

            # --- ضبط أبعاد المنظر والتفاعل ---
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='Inline / X', range=[0, nx]),
                    yaxis=dict(title='Crossline / Y', range=[0, ny]),
                    zaxis=dict(title='Depth / Time (Samples)', range=[0, nsamples], autorange='reversed'),
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                height=750
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"حدث خطأ أثناء العرض: {e}")
