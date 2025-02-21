#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/2/21 23:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :   

from streamlit import title, divider, expander, caption, empty, sidebar, markdown

title("Call a Ollama Model Locally")
divider()

empty_message: empty = empty()

with sidebar:
    "[![Static Badge](https://img.shields.io/badge/GitHub-py--st--ollama%20in%20DaoChaShao-black?style=for-the-badge&logo=github)](https://github.com/DaoChaShao/py-st-ollama)"

with expander("**TARGET**: Download the Ollama model and run it locally.", expanded=True):
    caption("1. Download the **INSTALLER** from [Ollama](https://ollama.com/).")
    caption("2. Download the **MODEL** you want to use.")
    caption("3. Keep the **INSTALLER** running.")
    caption("4. Clone the code from the **GitHub** to the local.")
    caption("5. Use the command `streamlit run main.py` to use this application.")

empty_message.info("You can call the Model at the **Models** page.")
