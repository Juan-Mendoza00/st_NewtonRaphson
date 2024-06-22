# Source Code for Streamlit app

## Newton-Raphson Calculator Inerface.

This app is deployed on Streamlit Community Cloud. Access here: https://st-newtonraphson.streamlit.app/

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Run this app locally](#Develop)
4. [Usage](#usage)

## Introduction

Note: ***Only for educational purposes***

Welcome to **Newton-Raphson Calculator**! This is an user-friendly application designed to easily compute roots of a *real-valued function* using the *Newton-Raphson* method for Numerical Analysis. Whether you're a developer (maybe) looking for some useful code snippet, go ahead and feel free to get some ideas from this implementation!. If you're a casual user (or a just a curious math student) in need of a really easy solution related to this matter, this App has you covered (almost always).

## Features
- **Simple and friendly interface:** The path is pretty straightforward, type your function expression making sure you're using allowed Python Syntax, click run and that's it 😊.
- **Display data from every iteration:** Already said that the implementation of Newton-Raphson here is a recursive funciton. From each iteration it computes an aproximation of the root and the *Relative Normalized Error*.
- **The Modified Formula:** The *Modified Newton-Raphson Method* is also implemented.
- **Responsive Design:** All the credit to [Streamlit](https://streamlit.io/) - it Works on both desktop and mobile devices (much better on PC).
- **Visualize:** At the bottom you can find a section to plot your function and spot the roots within an interval of your choice.

## Develop
### Run this app locally:

Prerequisites:
- [Python](https://www.python.org/). Version 3.8 to 3.12 supported
- The Linux Distribution you're comfortable with (this was developed in [Ubuntu](https://ubuntu.com/) running on [WSL2](https://ubuntu.com/desktop/wsl)). Everything can also be done in Windows through PowerShell.
- [pip](https://virtualenv.pypa.io/en/latest/) for package management and already comes wiht Python.
- A Python environment manager: in this project [virtualenv](https://virtualenv.pypa.io/en/latest/) is used.


### Steps
1. **Clone the repository:**
    ```bash
    git clone https://github.com/Juan-Mendoza00/st_NewtonRaphson.git
    cd st_NewtonRaphson
    ```

2. **Create the python environment:**
    ```bash
    python -m virtualenv env
    ```

3. **Activate the environment and install the dependencies with requirements file:**
    ```bash
    source env/bin/activate
    ```

    then, 

    ```bash
    pip install -r requirements.txt
    ```

4. **Start the application in localhost:**
    ```bash
    streamlit run my_app.py
    ```
    Follow the instructions to open the app in your browser and start using as you were navigating any website.