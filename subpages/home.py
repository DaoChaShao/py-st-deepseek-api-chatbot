#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/2/21 23:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :   

from streamlit import title, divider, expander, caption, empty, sidebar

title("Call the DeepSeek Model via API Key")
divider()

empty_message: empty = empty()

with sidebar:
    "[![Static Badge](https://img.shields.io/badge/GitHub-py--st--deepseek--api%20in%20DaoChaShao-black?style=for-the-badge&logo=github)](https://github.com/DaoChaShao/py-st-deepseek-api)"

with expander("**Instructions**: Call the deepseek model via API key.", expanded=True):
    caption("1. Clone the code from the [GitHub](https://github.com/DaoChaShao/py-st-deepseek-api) to the local.")
    caption("2. Run the command `pip install -r requirements.txt` to install the dependencies.")
    caption("3. Use the command `streamlit run main.py` to use this application.")
    caption("4. Apply for the API Key from [DeepSeek](https://deepseek.ai/).")
    caption("5. Enter the API Key in the **API Key** page.")

empty_message.info("You can call the Model at the **Models** page.")
