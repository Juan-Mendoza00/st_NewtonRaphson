import streamlit as st
import pandas as pd
import numpy as np
from sympy import *

from functions import Newton_Raphson, Newton_Raphson_modified, text_book_chart, stored_values, mdf_stored_values

x = Symbol('x')

# Title
st.title("Newton-Raphson's Method")
# a small body
st.markdown("""
            ### The Newton-Raphson method for numerical Analysis
            
            The Newton-Raphson (NR) method is used to *calculate an aproximation* of the **roots** of a *real-valued function* $f(x)$.
                    
            Newton-Raphson iterative method $\\rightarrow x_{i+1} = x_{i} - \\frac{f(x_{i})}{f'(x_{i})}$
                    
            Requiring $f$ to be differentiable and some **initial value**: $x_{0}$
        """)

st.divider()

# subheader
st.subheader('✍️ Write the function expression $f(x)$')

with st.container(border=True):
    
    col1, col2 = st.columns([0.6,0.4])

    with col1:
        expr = st.text_input('Write your function here. Make sure you use **Python Syntax**', value='x')
        
        # Sympyfy the text input
        func = sympify(expr).expand()
        st.write("**Visualize your function:** $f(x)$", func)

        # lamdify function and its derivates
        f = lambdify(x, func)
        df = lambdify(x, func.diff(x))
        df2 = lambdify(x,func.diff(x,2))
        
    with col2.container(border=True):
        st.markdown("**First derivate** $f'(x)$:")
        st.latex(func.diff(x).expand())

        st.markdown("**Second derivate** $f''(x)$:")
        st.latex(func.diff(x,x).expand())


# Implementation of the method
st.subheader('📝 Implementing the method')
st.write("$x_{i+1} = x_{i} - \\frac{f(x_{i})}{f'(x_{i})}$")
with st.container(border=True):
    st.write('⚙️ Parametter adjusting')
    param_col, data_col = st.columns([0.3,0.7]) 

# Parametters column
    with param_col.container(border=True):
        init_val = st.number_input(
            'Initial value $x_{0}$:',
            value=None,
            placeholder='int or float',
            key='NR_param init_val')
        
        iters = st.number_input(
            'Number of iterations:',
            value=None,
            placeholder='Integer',
            format='%.0f',
            key='NR_param iters')
        
        # Button to call tha function
        run = st.button('Run', type='primary', key='NR call')

# Data column
    with data_col.container(border=True):
        if run:
            # call the function
            value = Newton_Raphson(
                x_i=init_val,
                i=iters,
                f=f,
                df=df)
            
            # create a dataFrame
            dataframe1 = pd.DataFrame(
            data=[row[1:] for row in stored_values],
            index=[f'i = {row[0]}' for row in stored_values],
            columns=['x_i', 'Normalized Error']
            )
            # show the dataFrame
            st.dataframe(dataframe1, use_container_width=True, key='NR dataframe')
            # show the aproximation value
            st.write(f'- **There is a root in** $x = {value}$')
        else:
            st.caption('Data from every iteration of the method will be displayed here...')

on = st.toggle('Plot the function')
if on:

    # The function is passed into the plotting function
    fig = text_book_chart(
            func=f)
    
    st.pyplot(fig)


st.divider()
# Modified method section
st.subheader('💡 Modified Newton-Raphson Method')
st.write("The modified formula: $x_{i+1} = x_{i} - \\frac{f(x_{i})}{f'(x_{i})}$")
with st.container(border=True):
    param_col, data_col = st.columns([0.3,0.7]) 

# Parametters column
    with param_col.container(border=True):
        init_val = st.number_input(
            'Initial value $x_{0}$:',
            value=None,
            placeholder='int or float',
            key='M-NR_param init_val')
        
        iters = st.number_input(
            'Number of iterations:',
            value=None,
            placeholder='Integer',
            format='%.0f',
            key='M-NR_param iters')
        
        # Button to call tha function
        run = st.button('Run', type='secondary', key='M-NR call')

# Data column
    with data_col.container(border=True):
        if run:
            value = Newton_Raphson_modified(
                x_i=init_val,
                i=iters,
                f=f,
                df=df,
                df2=df2)
            

            dataframe2 = pd.DataFrame(
            data=[row[1:] for row in mdf_stored_values],
            index=[f'i = {row[0]}' for row in mdf_stored_values],
            columns=['x_i', 'Normalized Error']
            )
            st.dataframe(dataframe2, use_container_width=True, key='M-NR dataframe')


            st.write(f'- **There is a root in** $x = {value}$')
        else:
            st.caption('Data from every iteration of the method will be displayed here...')