import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from altair import datum

np.seterr(divide='ignore', invalid='ignore')

def top_hat(t, fwidth):
    N = len(t)
    y = np.zeros(N)
    i = np.where(np.abs(t) < fwidth)
    y[i] = 1.0
    return y

def ramp(t, fwidth):
    N = len(t)
    y = np.zeros(N)
    i = np.where(np.abs(t) < fwidth)
    y[i] = t[i]/fwidth
    return y

def delta(t, fwidth):
    N = len(t)
    y = np.zeros(N)
    i = np.where(t == float(fwidth))
    y[i] = 1.0
    return y

def get_cossin(data, u=5):
    t = data["t"]
    data["cos"] =  np.cos(2*np.pi*u/1000.0*t)
    data["sin"] = -np.sin(2*np.pi*u/1000.0*t)
    data["cos1"] = data["1"]*data["cos"]
    data["sin1"] = data["1"]*data["sin"]
    data["cos2"] = data["2"]*data["cos"]
    data["sin2"] = data["2"]*data["sin"]
    data["cos3"] = data["3"]*data["cos"]
    data["sin3"] = data["3"]*data["sin"]
    return data

def get_data(t, fwidth=50):
    data = pd.DataFrame(data={"t": t,
                              "1": top_hat(t, fwidth),
                              "2": ramp(t, fwidth),
                              "3": delta(t, fwidth)})
    return data

def get_fft(u, fwidth=50):
    v = 2.0*np.pi/1000.0*fwidth*u
    zeros = np.zeros(len(u))
    fft = pd.DataFrame(data={"u": u,
                             "real1": np.sin(2*np.pi/1000.0*fwidth*u)/(2*np.pi/1000.0*fwidth*u),
                             "imag1": zeros,
                             "real2": zeros,
                             "imag2": -2.0*(np.sin(v)-v*np.cos(v))/(v*v),
                             "real3": np.cos(v),
                             "imag3": np.sin(v)})
    fft["real1"] = fft["real1"].fillna(1.0)
    fft["imag2"] = fft["imag2"].fillna(0.0)
    return fft

t = np.linspace(-160, 160, 321)
N = len(t)
u = np.linspace(-80, 80, 161)
fwidth = 50

data = get_data(t, fwidth)
fft = get_fft(u, fwidth)

st.set_page_config(layout="wide")
st.title('Fourier transforms 1D')
col1, col2 = st.columns(2)

function = st.sidebar.selectbox("Select a function",
                            ["top hat", "ramp", "delta"],
                            index=0,)

component = st.sidebar.selectbox("Select component",
                             ["real", "imaginary"],
                             index=0,)

u = st.sidebar.slider('Select frequency', -80, 80, 5)
data = get_cossin(data, u=u)

data_dict = {"top hat": ["1", "cos1", "sin1", "real1", "imag1", "cos", "sin"],
             "ramp":    ["2", "cos2", "sin2", "real2", "imag2", "cos", "sin"],
             "delta":   ["3", "cos3", "sin3", "real3", "imag3", "cos", "sin"]}

cols = data_dict[function]
# pd.set_option('display.max_rows', 500)

ymin, ymax = (-1.1, 1.1)
p1 = alt.Chart(data, title="f(t)").mark_line().encode(x="t:Q",
                                                      y=alt.Y(cols[0], title="", scale=alt.Scale(domain=[ymin, ymax])))
if component == "real":
    p2 = alt.Chart(data, title="f(t)*cos(-2*pi*u*t)").mark_line().encode(x="t:Q",
                   y=alt.Y(cols[1], title="", scale=alt.Scale(domain=[ymin, ymax])))
    p3 = alt.Chart(data, title="cos(-2*pi*u*t)").mark_line().encode(x="t:Q",
                   y=alt.Y(cols[5], title="", scale=alt.Scale(domain=[ymin, ymax])))
    p4 = alt.Chart(fft, title="real part").mark_line().encode(x="u:Q",
                   y=alt.Y(cols[3], title="", scale=alt.Scale(domain=[ymin, ymax])))
    rule = alt.Chart(fft).mark_rule().encode(x="u:Q").transform_filter(datum.u == u)
else:
    p2 = alt.Chart(data, title="f(t)*sin(-2*pi*u*t)").mark_line().encode(x="t:Q",
                   y=alt.Y(cols[2], title="", scale=alt.Scale(domain=[ymin, ymax])))
    p3 = alt.Chart(data, title="sin(-2*pi*u*t)").mark_line().encode(x="t:Q",
                   y=alt.Y(cols[6], title="", scale=alt.Scale(domain=[ymin, ymax])))
    p4 = alt.Chart(fft, title="imag part").mark_line().encode(x="u:Q",
                   y=alt.Y(cols[4], title="", scale=alt.Scale(domain=[ymin, ymax])))
    rule = alt.Chart(fft).mark_rule().encode(x="u:Q").transform_filter(datum.u == u)

col1.altair_chart(p1, use_container_width=True)
col1.altair_chart(p2, use_container_width=True)
col2.altair_chart(p3, use_container_width=True)
col2.altair_chart(p4+rule, use_container_width=True)


# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)

# import plotly.graph_objects as go
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=data["t"], y=data[cols[0]]))
# fig.update_layout(title='f(t)',
#                   xaxis_title='t',
#                   yaxis_title='',
#                   width=800, height=800,
#                   margin=dict(l=40, r=40, b=40, t=40))
# st.plotly_chart(fig)
