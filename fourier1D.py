import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

def top_hat(x, fwidth):
    y = np.zeros(N)
    i = np.where(np.abs(x) < fwidth)
    y[i] = 1.0
    return y

def ramp(x, fwidth):
    y = np.zeros(N)
    i = np.where(np.abs(x) < fwidth)
    y[i] = x[i]/fwidth
    return y

def delta(x, fwidth):
    y = np.zeros(N)
    i = np.where(x == float(fwidth))
    y[i] = 1.0
    return y

def get_cossin(data, u=5):
    data["cos"] =  np.cos(2*np.pi*u/1000.0*x)
    data["sin"] = -np.sin(2*np.pi*u/1000.0*x)
    data["cos1"] = data["1"]*data["cos"]
    data["sin1"] = data["1"]*data["sin"]
    data["cos2"] = data["2"]*data["cos"]
    data["sin2"] = data["2"]*data["sin"]
    data["cos3"] = data["3"]*data["cos"]
    data["sin3"] = data["3"]*data["sin"]
    return data

def get_data(x, fwidth=50):
    data = pd.DataFrame(data={"x": x,
                              "1": top_hat(x, fwidth),
                              "2": ramp(x, fwidth),
                              "3": delta(x, fwidth)})
    return data

def get_fft(xi, fwidth=50):
    v = 2.0*np.pi/1000.0*fwidth*xi
    zeros = np.zeros(len(xi))
    fft = pd.DataFrame(data={"x": xi,
                             "real1": np.sin(2*np.pi/1000.0*fwidth*xi)/(2*np.pi/1000.0*fwidth*xi),
                             "imag1": zeros,
                             "real2": zeros,
                             "imag2": -2.0*(np.sin(v)-v*np.cos(v))/(v*v),
                             "real3": np.cos(v),
                             "imag3": np.sin(v)})
    fft["real1"] = fft["real1"].fillna(1.0)
    fft["imag2"] = fft["imag2"].fillna(0.0)
    return fft

x = np.linspace(-160, 160, 321)
N = len(x)
xi = np.linspace(-80, 80, 161)
fwidth = 50

data = get_data(x, fwidth)
fft = get_fft(xi, fwidth)

st.set_page_config(layout="wide")
st.title('Fourier transforms 1D')
choice, col1, col2 = st.columns([1, 3, 3])

function = choice.selectbox("Select a function",
                            ["top hat", "ramp", "delta"],
                            index=0,)

component = choice.selectbox("Select component",
                             ["real", "imaginary"],
                             index=0,)

u = 5.0
data = get_cossin(data, u=u)

function = "top hat"
data_dict = {"top hat": ["1", "cos1", "sin1", "real1", "imag1", "cos", "sin"],
             "ramp":    ["2", "cos2", "sin2", "real2", "imag2", "cos", "sin"],
             "delta":   ["3", "cos3", "sin3", "real3", "imag3", "cos", "sin"]}

cols = data_dict[function]
pd.set_option('display.max_rows', 500)
pdb.set_trace()

ymin, ymax = (-1.1, 1.1)
p1 = alt.Chart(data).mark_line().encode(x="x", y=alt.Y(cols[0], scale=alt.Scale(domain=[ymin, ymax])))
if component == "real":
    p2 = alt.Chart(data).mark_line().encode(x="x", y=alt.Y(cols[1], scale=alt.Scale(domain=[ymin, ymax])))
    p3 = alt.Chart(data).mark_line().encode(x="x", y=alt.Y(cols[5], scale=alt.Scale(domain=[ymin, ymax])))
    p4 = alt.Chart(fft).mark_line().encode(x="x",  y=alt.Y(cols[3], scale=alt.Scale(domain=[ymin, ymax])))
else:
    p2 = alt.Chart(data).mark_line().encode(x="x", y=alt.Y(cols[2], scale=alt.Scale(domain=[ymin, ymax])))
    p3 = alt.Chart(data).mark_line().encode(x="x", y=alt.Y(cols[6], scale=alt.Scale(domain=[ymin, ymax])))
    p4 = alt.Chart(fft).mark_line().encode(x="x",  y=alt.Y(cols[4], scale=alt.Scale(domain=[ymin, ymax])))

# col1.altair_chart(p1, use_container_width=True)
# col1.altair_chart(p2, use_container_width=True)
# col2.altair_chart(p3, use_container_width=True)
# col2.altair_chart(p4, use_container_width=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
