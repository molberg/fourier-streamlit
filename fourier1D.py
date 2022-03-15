import numpy as np
import streamlit as st
import plotly.graph_objects as go

np.seterr(divide="ignore", invalid="ignore")


@st.cache(suppress_st_warning=True)
def top_hat(t, fwidth):
    N = len(t)
    y = np.zeros(N)
    i = np.where(np.abs(t) < fwidth)
    y[i] = 1.0
    return y


@st.cache(suppress_st_warning=True)
def ramp(t, fwidth):
    N = len(t)
    y = np.zeros(N)
    i = np.where(np.abs(t) < fwidth)
    y[i] = t[i] / fwidth
    return y


@st.cache(suppress_st_warning=True)
def delta(t, fwidth):
    N = len(t)
    y = np.zeros(N)
    i = np.where(t == float(fwidth))
    y[i] = 1.0
    return y


@st.cache(suppress_st_warning=True)
def get_cos1D(t, f):
    y = np.cos(2 * np.pi * f * t)
    return y


@st.cache(suppress_st_warning=True)
def get_sin1D(t, f):
    y = -np.sin(2 * np.pi * f * t)
    return y


@st.cache(suppress_st_warning=True)
def get_sin1D(t, f):
    y = -np.sin(2 * np.pi * f * t)
    return y


@st.cache(suppress_st_warning=True)
def get_fcos1D(ft, t, f):
    fy = ft * get_cos1D(t, f)
    return fy


@st.cache(suppress_st_warning=True)
def get_fsin1D(ft, t, f):
    fy = ft * get_sin1D(t, f)
    return fy


@st.cache(suppress_st_warning=True)
def get_fft1D(model, part, u, fwidth):
    v = 2.0 * np.pi * fwidth * u
    zeros = np.zeros(len(u))
    if model == "top hat":
        if part == "real":
            fft = np.sin(2 * np.pi * fwidth * u) / (
                2 * np.pi * fwidth * u
            )
            fft[np.isnan(fft)] = 1.0
        else:
            fft = zeros
    elif model == "ramp":
        if part == "real":
            fft = zeros
        else:
            fft = -2.0 * (np.sin(v) - v * np.cos(v)) / (v * v)
            # i = np.where(np.isnan(fft))
            fft[np.isnan(fft)] = 0.0
    elif model == "delta":
        if part == "real":
            fft = np.cos(v)
        else:
            fft = np.sin(v)
    return fft

# t is spatial dimension in mm
t = np.linspace(-1.6, 1.6-0.0125, 256)
u = np.fft.fftshift(np.fft.fftfreq(t.size, d=0.0125))
fwidth = 0.5

st.set_page_config(layout="wide")
st.title("Fourier transforms 1D")

model = st.sidebar.selectbox("Select a function", ["top hat", "ramp", "delta"], index=0)
part = st.sidebar.selectbox("Select component", ["real", "imaginary"], index=0)
f = st.sidebar.slider("Select u-frequency", -5.0, 5.0, value=1.0, step=0.25)
# f = nf*0.03125

col1, col2 = st.columns(2)

if model == "top hat":
    ft = top_hat(t, fwidth)
elif model == "ramp":
    ft = ramp(t, fwidth)
elif model == "delta":
    ft = delta(t, fwidth)

if part == "real":
    trig = get_cos1D(t, f)
    ftrig = get_fcos1D(ft, t, f)
else:
    trig = get_sin1D(t, f)
    ftrig = get_fsin1D(ft, t, f)

fft = get_fft1D(model, part, u, fwidth)
trig_label = {
    "real": "cos(-2\u03c0\u00b7u\u00b7x)",
    "imaginary": "sin(-2\u03c0\u00b7u\u00b7x)",
}
ftrig_label = {
    "real": "f(x)\u00b7cos(-2\u03c0\u00b7u\u00b7x)",
    "imaginary": "f(x)\u00b7sin(-2\u03c0\u00b7u\u00b7x)",
}

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=t, y=ft))
fig1.update_layout(
    title="f(x)",
    xaxis_title="x [cm]",
    yaxis_title="",
    height=300,
    margin=dict(l=0, r=40, t=30, b=20),
)
col1.plotly_chart(fig1, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=t, y=trig))
fig2.update_layout(
    title=trig_label[part],
    xaxis_title="x [cm]",
    yaxis_title="",
    height=300,
    margin=dict(l=0, r=40, t=30, b=20),
)
col2.plotly_chart(fig2, use_container_width=True)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=t, y=ftrig, fill="tozeroy"))
fig3.update_layout(
    title=ftrig_label[part],
    xaxis_title="x [cm]",
    yaxis_title="",
    height=300,
    margin=dict(l=0, r=40, t=30, b=20),
)
col1.plotly_chart(fig3, use_container_width=True)

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=u, y=fft))
fig4.add_vline(x=f, line_width=1, line_color="green")
fig4.update_layout(
    title=part + " part",
    xaxis_title="u [cycles/cm]",
    yaxis_title="",
    height=300,
    margin=dict(l=0, r=40, t=30, b=20),
)
col2.plotly_chart(fig4, use_container_width=True)

# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)
