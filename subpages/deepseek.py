#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/2/21 22:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   deepseek.py
# @Desc     :   

from streamlit import (chat_input, empty, chat_message, write, sidebar,
                       session_state, button, caption, rerun)

from utilis.tools import parameters, model_caller, Timer, api_key_checker

# Initialize the chat history
if "messages" not in session_state:
    session_state.messages = []

empty_message: empty = empty()

model, api_key, sys_content, temperature, top_p = parameters()
prompt: str = chat_input("Say something", max_chars=100)

if model != "Select a Model":

    # Display the chat history firstly
    for msg in session_state.messages:
        with chat_message(msg["role"]):
            write(msg["content"])

    if api_key:
        if api_key_checker(api_key):
            empty_message.success("The API key has been set.")
            if prompt:
                # Add the user input to the chat history
                session_state.messages.append({"role": "user", "content": prompt})
                # Display the user input
                with chat_message("user"):
                    write(prompt)

                # Create a full prompt with the conversation history
                prompt_all = "\n".join([msg["content"] for msg in session_state.messages])

                with Timer(2, description="Local Model Call") as timer:
                    with chat_message("assistant"):
                        response = model_caller(model, api_key, temperature, top_p, sys_content, prompt_all)  # 生成 AI 回复
                        write(response)

                        # Add the AI response to the chat history
                        session_state.messages.append({"role": "assistant", "content": response})

                empty_message.success(f"**{str(timer)}**")
            else:
                empty_message.error("Please enter a prompt to generate a response.")
    else:
        empty_message.error("Please enter the API key for the model.")
else:
    empty_message.info("Please select a model from the sidebar.")

with sidebar:
    caption(f"**{int(len(session_state.messages) / 2)}** round messages in the chat history.")
    if button("Clear History", type="primary", help="Click to clear the chat history."):
        session_state.messages = []
        empty_message.success("The history has been cleared.")
        rerun()
