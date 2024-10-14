import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)

st.title("波兰球 —— 高山下的花环")
st.image("./data/image/poland_ball.jpg")

st.title("data frame load test")


@st.cache_data
def load_data():
    return pd.read_csv("./data/uncleaned/books-kaggle-mohamadreza-momeni.csv", on_bad_lines='skip')


df = load_data()
st.dataframe(df.head(10))
