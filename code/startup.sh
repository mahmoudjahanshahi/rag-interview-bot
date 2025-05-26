#!/bin/bash

# This script is used to set up the environment for a Streamlit application.
# It installs the required packages and starts the Streamlit server.
pip install -r requirements.txt

exec streamlit run app.py --server.port=$PORT --server.enableCORS=false
