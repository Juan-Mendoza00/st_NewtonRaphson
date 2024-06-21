import streamlit as st
import pandas as pd
import numpy as np
from sympy import *

from analysis import NewtonRaphson, text_book_chart

# Using wide mode
st.set_page_config(
    page_title='Newton-Raphson Computer',
    layout='wide'
)

# Creating session variables
if 'NR_clicked' not in st.session_state:
    st.session_state['NR_clicked'] = False
    st.session_state['NR_data'] = {}
    st.session_state['M-NR_clicked'] = False
    st.session_state['M-NR_data'] = {}


def NR_run():
    st.session_state['NR_clicked'] = True

def MNR_run():
    st.session_state['M-NR_clicked'] = True


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

# --- "NORMAL METHOD" SECTION
st.subheader('✍️ Type the function expression $f(x)$')

with st.container(border=True):
    
    col1, col2 = st.columns([0.5,0.5])

    with col1:
        expr = st.text_input('Write your function here. Make sure you use **Python Syntax**', value='x')
        
        # Sympyfy the text input
        func = sympify(expr, rational=True).expand()
        # get latex representation
        func_latex = latex(func)

        st.latex(f"f(x) = {func_latex}")

    with col2.container(border=True):
        st.markdown("**First and Second Derivates:**")
        st.latex(f"f'(x) = {latex(func.diff(x).expand())}")

        st.latex(f"f''(x) = {latex(func.diff(x,x).expand())}")


# Implementation of the method
st.subheader('📝 Implementing the method')
st.write("$x_{i+1} = x_{i} - \\frac{f(x_{i})}{f'(x_{i})}$")

with st.container(border=True):
    param_col, data_col = st.columns([0.3,0.7]) 

    # Parametters column
    with param_col.container(border=True):
        st.write('⚙️ **Parametter adjusting**')
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
        run = st.button('Run', type='primary', key='NR call', on_click=NR_run)
        if run:
            # Instance the computer class
            NR = NewtonRaphson(function_expr=func, modified=False)
            # call the function
            value = NR.compute_root(
                x_i=init_val,
                i=iters)
            
            # create a dataFrame
            dataframe1 = pd.DataFrame(
            data=[row[1:] for row in NR.stored_aproximations],
            index=[row[0] for row in NR.stored_aproximations],
            columns=['x_i', 'Normalized Error (%)']
            )

            # save dataframe and value to session state variables
            st.session_state['NR_data']['dataframe'] = dataframe1
            st.session_state['NR_data']['aprox'] = value

    # Data column
    with data_col.container(border=True):        
        if st.session_state['NR_clicked']:
            # show the dataFrame
            st.dataframe(st.session_state['NR_data']['dataframe'], 
                         use_container_width=True, 
                         key='NR dataframe')

            # show the aprox value
            st.write(f"- **Computed root** $$x = {st.session_state['NR_data']['aprox']}$$")
        else:
            st.caption('Data from every iteration of the method will be displayed here...')


st.divider()

# --- MODIFIED METHOD SECTION
st.subheader('💡 Modified Newton-Raphson Method')
st.write("The modified formula: $x_{i+1} = x_{i} - \\frac{f(x_{i})f'(x_{i})}{f'(x_{i})^{2} - f(x_{i})f''(x_{i})}$")
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
        run = st.button('Run', type='secondary', key='M-NR call', on_click=MNR_run)
        if run:
            M_NR = NewtonRaphson(function_expr=func, modified=True)
            # call the function
            value = M_NR.compute_root(
                x_i=init_val,
                i=iters)
            
            # build the dataframe
            dataframe2 = pd.DataFrame(
                data=[row[1:] for row in M_NR.stored_aproximations],
                index=[row[0] for row in M_NR.stored_aproximations],
                columns=['x_i', 'Normalized Error (%)']
            )

            # modify session variables
            st.session_state['M-NR_data']['dataframe'] = dataframe2
            st.session_state['M-NR_data']['aprox'] = value

    # Data column
    with data_col.container(border=True):
        if st.session_state['M-NR_clicked']:
            # show the dataframe
            st.dataframe(st.session_state['M-NR_data']['dataframe'], 
                         use_container_width=True, 
                         key='M-NR dataframe')
            
            # Show the aproximation
            st.write(f'- **There is a root in** $x = {st.session_state["M-NR_data"]["aprox"]}$')
        else:
            st.caption('Data from every iteration of the method will be displayed here...')

st.divider()

# --- PLOTTING SECTION
st.subheader('Visualize the function',
             help='Visually spot the root of the function')

if "figure" not in st.session_state:
    st.session_state.figure = None

params, graph = st.columns([0.2,0.8])

# Interval choosing column
with params:
    st.write('Choose the X-axis interval')
    x_lower = st.number_input('Lower bound', value=None, placeholder='x = ')
    x_upper = st.number_input('Upper bound', value=None, placeholder='x = ')

    plot_graph = st.button('Show')
    if plot_graph: 
        # Passing function and interval for x variable
        fig = text_book_chart(
                f=lambdify(x, func),
                interval=(x_lower,x_upper))
        st.session_state.figure = fig
        
    expression_latex = f"f(x) = {latex(func)}"
    st.latex(expression_latex)
    st.latex(f"x \in I = [{x_lower}, {x_upper}]")

# Chart column
with graph.container(border=True):

    place_holder = st.empty()

    if st.session_state.figure is None:
        place_holder.caption(
            """Choose values for Lower and Upper bounds of the interval in which you want to visualize the function.
            Click 'Show' button and you'll see it. """
        )
    else:
        place_holder.pyplot(st.session_state.figure)