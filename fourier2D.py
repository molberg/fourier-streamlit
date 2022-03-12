import numpy as np
import streamlit as st
import plotly.graph_objects as go

np.seterr(divide="ignore", invalid="ignore")


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
st.title("Fourier transforms 2D")

model = st.sidebar.selectbox("Select a function", ["circular", "rectangular"], index=0)
part = st.sidebar.selectbox("Select component", ["real", "imaginary"], index=0)
f1 = st.sidebar.slider("Select u-frequency", -80, 80, 5)
f2 = st.sidebar.slider("Select v-frequency", -80, 80, 5)
xwidth = st.sidebar.slider("Select width", 0, 160, 50)

if model == "rectangular":
    ywidth = st.sidebar.slider("Select 2nd width", 0, 160, 50)
    tilt = st.sidebar.slider("rotate rectangle", 0, 180, 0)

col1, col2 = st.columns(2)

if model == "circular":
    ft = circular(t, xwidth)
else:
    ft = rectangular(t, xwidth, ywidth, tilt)

fft = get_fft2D(model, xwidth, ywidth, tilt)

if part == "real":
    trig = get_cos2D(t, f1, f2)
    ftrig = get_fcos2D(ft, t, f1, f2)
    fftpart = np.real(fft)
else:
    trig = get_sin2D(t, f1, f2)
    ftrig = get_fsin2D(ft, t, f1, f2)
    fftpart = np.imag(fft)

fig1 = go.Figure(data=[go.Surface(z=ft, x=t, y=t, showscale=False)])
fig1.update_layout(title="", margin=dict(l=40, r=40, b=40, t=40))
col1.plotly_chart(fig1, use_container_width=True)

fig2 = go.Figure(data=[go.Surface(z=trig, x=t, y=t, showscale=False)])
fig2.update_layout(title="", margin=dict(l=40, r=40, b=40, t=40))
col2.plotly_chart(fig2, use_container_width=True)

fig3 = go.Figure(data=[go.Surface(z=ftrig, x=t, y=t, showscale=False)])
fig3.update_layout(title="", margin=dict(l=40, r=40, b=40, t=40))
col1.plotly_chart(fig3, use_container_width=True)

fig4 = go.Figure(data=[go.Surface(z=fftpart, x=u, y=u, showscale=False)])
i = np.where(u == f1)
j = np.where(u == f2)
fig4.add_trace(
    go.Scatter3d(x=[f1], y=[f2], z=[fftpart[80 + f1, 80 + f2]], mode="markers")
)
fig4.update_traces(marker_size=5, selector=dict(type="scatter3d"))
fig4.update_traces(marker_color="green", selector=dict(type="scatter3d"))
fig4.update_layout(title="", margin=dict(l=40, r=40, b=40, t=40))
col2.plotly_chart(fig4, use_container_width=True)

# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)
