"""
Simple hello world Streamlit app to test infrastructure.
This is a minimal app to verify Docker and Streamlit are running correctly.
"""
import streamlit as st

st.set_page_config(page_title="MNIST Infrastructure Check", page_icon="🔍")

st.title("Hello Infrastructure!")
st.write("This is just to confirm Docker and Streamlit are running.")

st.success("✅ If you can see this, your basic infrastructure is working!")

# Add some simple interactive elements to test Streamlit functionality
if st.checkbox("Show more details"):
    st.write("This is a placeholder for the MNIST digit classifier app.")
    st.info("The complete app will include a model to classify handwritten digits.")
    
    st.subheader("Next steps:")
    st.markdown("""
    1. Train the MNIST model
    2. Implement the drawing/upload interface
    3. Connect to the PostgreSQL database
    4. Deploy the full application
    """)

st.sidebar.header("Infrastructure Test")
st.sidebar.write("Everything seems to be working properly!") 