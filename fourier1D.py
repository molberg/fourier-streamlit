import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(layout="wide")
st.title('Fourier transforms 1D')
choice, col1, col2 = st.columns([1, 3, 3])

function = choice.selectbox("Select a function",
                            ["top hat", "ramp", "delta"],
                            index=0,)

component = choice.selectbox("Select component",
                             ["real", "imaginary"],
                             index=0,)

x = np.linspace(-160, 160, 321)
xi = np.linspace(-80, 80, 161)
fwidth = 50
u = 5.0

v = 2.0*np.pi/1000.0*fwidth*xi

cos = np.cos(2*np.pi*u/1000.0*x)
sin = -np.sin(2*np.pi*u/1000.0*x)

N = len(x)

top_hat = np.zeros(N)
i = np.where(np.abs(x) < fwidth)
top_hat[i] = 1.0

ramp = np.zeros(N)
i = np.where(np.abs(x) < fwidth)
ramp[i] = x[i]/fwidth

delta = np.zeros(N)
i = np.where(x == float(fwidth))
delta[i] = 1.0

if function == "top hat":
    data = pd.DataFrame(data={"x": x, "fun": top_hat, "cos": cos, "sin": sin})
    fft = pd.DataFrame(data={"x": xi,
                             "real": np.sin(2*np.pi/1000.0*fwidth*xi)/(2*np.pi/1000.0*fwidth*xi),
                             "imag": np.zeros(len(xi))})
    fft["real"] = fft["real"].fillna(1.0)
elif function == "ramp":
    data = pd.DataFrame(data={"x": x, "fun": ramp, "cos": cos, "sin": sin})
    fft = pd.DataFrame(data={"x": xi,
                             "real": np.zeros(len(xi)),
                             "imag": -2.0*(np.sin(v)-v*np.cos(v))/(v*v)})
    fft["imag"] = fft["imag"].fillna(0.0)
else:
    data = pd.DataFrame(data={"x": x, "fun": delta, "cos": cos, "sin": sin})
    fft = pd.DataFrame(data={"x": xi,
                             "real": np.cos(v),
                             "imag": np.sin(v)})

data["fcos"] = data["fun"]*cos
data["fsin"] = data["fun"]*sin

p1 = alt.Chart(data).mark_line().encode(x="x", y=alt.Y("fun", scale=alt.Scale(domain=[-1.1, 1.1])))
if component == "real":
    p2 = alt.Chart(data).mark_line().encode(x="x", y=alt.Y("fcos", scale=alt.Scale(domain=[-1.1, 1.1])))
    p3 = alt.Chart(data).mark_line().encode(x="x", y=alt.Y("cos",  scale=alt.Scale(domain=[-1.1, 1.1])))
    p4 = alt.Chart(fft).mark_line().encode(x="x",  y=alt.Y("real", scale=alt.Scale(domain=[-1.1, 1.1])))
else:
    p2 = alt.Chart(data).mark_line().encode(x="x", y=alt.Y("fsin", scale=alt.Scale(domain=[-1.1, 1.1])))
    p3 = alt.Chart(data).mark_line().encode(x="x", y=alt.Y("sin",  scale=alt.Scale(domain=[-1.1, 1.1])))
    p4 = alt.Chart(fft).mark_line().encode(x="x",  y=alt.Y("imag", scale=alt.Scale(domain=[-1.1, 1.1])))

col1.altair_chart(p1, use_container_width=True)
col1.altair_chart(p2, use_container_width=True)
col2.altair_chart(p3, use_container_width=True)
col2.altair_chart(p4, use_container_width=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
