#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/2/21 22:58
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   tools.py
# @Desc     :   

from openai import OpenAI
from streamlit import (sidebar, header, selectbox, caption, text_input,
                       segmented_control, slider, empty)
from time import perf_counter


def parameters() -> tuple[str, str, str, float, float]:
    """ Set the hyperparameters for the Ollama model.

    :return: the model name, temperature, and top P
    """
    content: str = ""
    temperature: float = 0.0
    param_temp: float = 0.0
    top_p: float = 0.0

    with sidebar:
        header("Hyperparameters")

        options: list = ["deepseek-chat"]
        model: str = selectbox(
            "Model", options, disabled=True, help="Select a model"
        )
        caption(f"The model you selected is: **{model}**")

        api_key: str = text_input(
            "API Key", placeholder="Enter your API key of the deepseek", type="password",
            max_chars=35, help="The API key for the model."
        )
        caption(f"The length of the API key entered is: **{len(api_key)}**")

        if api_key:
            controls: list = ["General", "Math/Code", "Translation"]
            category: str = segmented_control(
                "Content", controls, default=controls[0], selection_mode="single",
                help="The instruction of the system role."
            )
            match category:
                case "General":
                    content = "The system will generate a general response."
                    param_temp = 1.3
                case "Math/Code":
                    content = "The system will generate a response related to math or code."
                    param_temp = 0.0
                case "Translation":
                    content = "The system will generate a translation response."
                    param_temp = 1.5

            temperature: int = slider("Temperature", 0.0, 2.0, param_temp, help="The randomness of the output.")
            caption(f"The temperature you selected is: **{temperature}**")

            top_p: int = slider("Top P", 0.0, 1.0, 0.9, help="The probability of the output.")
            caption(f"The top P you selected is: **{top_p}**")

        return model, api_key, content, temperature, top_p


def model_caller(model: str, api_key: str, temperature: float, top_p: int, content: str, prompt: str) -> str:
    """ Call the Ollama model locally via requests package.

    :param model: the model name
    :param api_key: the API key for the model
    :param temperature: the randomness of the output
    :param top_p: the probability of the output
    :param content: the instruction of the system role
    :param prompt: the input from the user
    :return: the response from the model
    """
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        stream=False,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def api_key_checker(api_key: str) -> bool:
    """ Check the API key format.

    :param api_key: enter the API key of the deepseek
    :return: True if the API key is valid
    """
    if api_key.startswith("sk-") and len(api_key) == 35:
        return True


class Timer(object):
    """ A simple timer class to measure the elapsed time.

    :param precision: the number of decimal places to round the elapsed time
    :param description: the description of the timer
    """

    def __init__(self, precision: int = 5, description: str = None):
        self._precision = precision
        self._description = description

    def __enter__(self):
        self._start = perf_counter()
        return self

    def __exit__(self, *args):
        self._end = perf_counter()
        self._elapsed = self._end - self._start

    def __repr__(self):
        return f"{self._description} took {self._elapsed:.{self._precision}f} seconds."
