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
def circular(t, fwidth=50):
    N = len(t)
    X, Y = np.meshgrid(t, t)
    Z = np.zeros((N, N))
    i = np.where(np.abs(X * X + Y * Y) < fwidth * fwidth)
    Z[i] = 1.0
    return Z


@st.cache(suppress_st_warning=True)
def rectangular(t, xwidth=50, ywidth=50, tilt=0.0):
    N = len(t)
    X, Y = np.meshgrid(t, t)
    Z = np.zeros((N, N))
    rad = tilt * np.pi / 180.0
    XR = X * np.cos(rad) + Y * np.sin(rad)
    YR = -X * np.sin(rad) + Y * np.cos(rad)
    i = np.logical_and(abs(XR) < xwidth, np.abs(YR) < ywidth)
    Z[i] = 1.0
    return Z


@st.cache(suppress_st_warning=True)
def get_cos1D(t, f):
    y = np.cos(2 * np.pi * f / 1000.0 * t)
    return y


@st.cache(suppress_st_warning=True)
def get_sin1D(t, f):
    y = -np.sin(2 * np.pi * f / 1000.0 * t)
    return y


@st.cache(suppress_st_warning=True)
def get_sin1D(t, f):
    y = -np.sin(2 * np.pi * f / 1000.0 * t)
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
def get_cos2D(t, u, v):
    N = len(t)
    X, Y = np.meshgrid(t, t)
    Z = np.cos(2 * np.pi * (u * X - v * Y) / N)
    return Z


@st.cache(suppress_st_warning=True)
def get_sin2D(t, u, v):
    N = len(t)
    X, Y = np.meshgrid(t, t)
    Z = np.sin(2 * np.pi * (u * X - v * Y) / N)
    return Z


@st.cache(suppress_st_warning=True)
def get_fcos2D(ft, t, u, v):
    N = len(t)
    X, Y = np.meshgrid(t, t)
    Z = ft * get_cos2D(t, u, v)
    return Z


@st.cache(suppress_st_warning=True)
def get_fsin2D(ft, t, u, v):
    N = len(t)
    X, Y = np.meshgrid(t, t)
    Z = ft * get_sin2D(t, u, v)
    return Z


@st.cache(suppress_st_warning=True)
def get_fft1D(model, part, u, fwidth):
    v = 2.0 * np.pi / 1000.0 * fwidth * u
    zeros = np.zeros(len(u))
    if model == "top hat":
        if part == "real":
            fft = np.sin(2 * np.pi / 1000.0 * fwidth * u) / (
                2 * np.pi / 1000.0 * fwidth * u
            )
            # i = np.where(np.isnan(fft))
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


@st.cache(suppress_st_warning=True)
def get_fft2D(model, xwidth, ywidth, tilt):
    if model == "circular":
        ft = circular(t, xwidth)
    else:
        ft = rectangular(t, xwidth, ywidth, tilt)
    Z = np.fft.fftshift(np.fft.fft2(ft))
    return Z[80:241, 80:241]


t = np.linspace(-160, 160, 321)
u = np.linspace(-80, 80, 161)
xwidth = 50
ywidth = 50
tilt = 0.0

st.set_page_config(layout="wide")
st.sidebar.title("Fourier transforms")

dim = st.sidebar.selectbox("Select dimension", ["1D", "2D"], index=0)
if dim == "1D":
    model = st.sidebar.selectbox(
        "Select function", ["top hat", "ramp", "delta"], index=0
    )
else:
    model = st.sidebar.selectbox(
        "Select a function", ["circular", "rectangular"], index=0
    )
part = st.sidebar.selectbox("Select component", ["real", "imaginary"], index=0)
f1 = st.sidebar.slider("Select u-frequency", -80, 80, 5)
if dim == "2D":
    f2 = st.sidebar.slider("Select v-frequency", -80, 80, 5)

if dim == "2D" and model == "rectangular":
    xwidth = st.sidebar.slider("Select x-width", 0, 160, 50)
    ywidth = st.sidebar.slider("Select y-width", 0, 160, 50)
    tilt = st.sidebar.slider("rotate rectangle", 0, 180, 0)
else:
    xwidth = st.sidebar.slider("Select width", 0, 160, 50)

col1, col2 = st.columns(2)

if dim == "1D":
    if model == "top hat":
        ft = top_hat(t, xwidth)
    elif model == "ramp":
        ft = ramp(t, xwidth)
    elif model == "delta":
        ft = delta(t, xwidth)

    if part == "real":
        trig = get_cos1D(t, f1)
        ftrig = get_fcos1D(ft, t, f1)
    else:
        trig = get_sin1D(t, f1)
        ftrig = get_fsin1D(ft, t, f1)

    fft = get_fft1D(model, part, u, xwidth)
    trig_label = {
        "real": "cos(-2\u03c0\u00b7u\u00b7t)",
        "imaginary": "sin(-2\u03c0\u00b7u\u00b7t)",
    }
    ftrig_label = {
        "real": "f(t)\u00b7cos(-2\u03c0\u00b7u\u00b7t)",
        "imaginary": "f(t)\u00b7sin(-2\u03c0\u00b7u\u00b7t)",
    }

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=t, y=ft))
    fig1.update_layout(
        title="f(t)",
        xaxis_title="t",
        yaxis_title="",
        height=300,
        margin=dict(l=0, r=40, t=30, b=20),
    )
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=trig))
    fig2.update_layout(
        title=trig_label[part],
        xaxis_title="t",
        yaxis_title="",
        height=300,
        margin=dict(l=0, r=40, t=30, b=20),
    )
    col2.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=t, y=ftrig, fill="tozeroy"))
    fig3.update_layout(
        title=ftrig_label[part],
        xaxis_title="t",
        yaxis_title="",
        height=300,
        margin=dict(l=0, r=40, t=30, b=20),
    )
    col1.plotly_chart(fig3, use_container_width=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=u, y=fft))
    fig4.add_vline(x=f1, line_width=1, line_color="green")
    fig4.update_layout(
        title=part + " part of fft",
        xaxis_title="u",
        yaxis_title="",
        height=300,
        margin=dict(l=0, r=40, t=30, b=20),
    )
    col2.plotly_chart(fig4, use_container_width=True)

else:
    if model == "circular":
        ft = circular(t, xwidth)
    else:
        ft = rectangular(t, xwidth, ywidth, tilt)

    fft = get_fft2D(model, xwidth, ywidth, tilt)
    trig_label = {
        "real": "cos(2\u03c0\u00b7(u\u00b7x-v\u00b7y))",
        "imaginary": "sin(2\u03c0\u00b7(u\u00b7x-v\u00b7y))",
    }
    ftrig_label = {
        "real": "f(x,y)\u00b7cos(2\u03c0\u00b7(u\u00b7x-v\u00b7y))",
        "imaginary": "f(x,y)\u00b7sin(2\u03c0\u00b7(u\u00b7x-v\u00b7y))",
    }

    if part == "real":
        trig = get_cos2D(t, f1, f2)
        ftrig = get_fcos2D(ft, t, f1, f2)
        fftpart = np.real(fft)
    else:
        trig = get_sin2D(t, f1, f2)
        ftrig = get_fsin2D(ft, t, f1, f2)
        fftpart = np.imag(fft)

    fig1 = go.Figure(data=[go.Surface(z=ft, x=t, y=t, showscale=False)])
    fig1.update_layout(title="f(x,y)", margin=dict(l=0, r=40, t=30, b=20))
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure(data=[go.Surface(z=trig, x=t, y=t, showscale=False)])
    fig2.update_layout(title=trig_label[part], margin=dict(l=0, r=40, t=30, b=20))
    col2.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure(data=[go.Surface(z=ftrig, x=t, y=t, showscale=False)])
    fig3.update_layout(title=ftrig_label[part], margin=dict(l=0, r=40, t=30, b=20))
    col1.plotly_chart(fig3, use_container_width=True)

    fig4 = go.Figure(data=[go.Surface(z=fftpart, x=u, y=u, showscale=False)])
    i = np.where(u == f1)
    j = np.where(u == f2)
    fig4.add_trace(
        go.Scatter3d(x=[f1], y=[f2], z=[fftpart[80 + f1, 80 + f2]], mode="markers")
    )
    fig4.update_traces(marker_size=5, selector=dict(type="scatter3d"))
    fig4.update_traces(marker_color="green", selector=dict(type="scatter3d"))
    fig4.update_layout(title=part + " part of fft", margin=dict(l=0, r=40, t=30, b=20))
    col2.plotly_chart(fig4, use_container_width=True)

# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)
