import streamlit as st
import pandas as pd
import numpy as np
from sympy import *

from analysis import NewtonRaphson, text_book_chart

# Using wide mode
st.set_page_config(
    page_title='Newton-Raphson Computer',
    # layout='wide'
)

with st.sidebar:
    st.title("📍Navigate")
    st.markdown(
        """
        ## Newton-Raphson's method for Numerical Analysis

        - [The Newton-Raphson Method](#newton-raphson-s-method)
        
        - [How it works](#how-it-works)

        - [Advantages, Disadvantages and Conclusion](#advantages)

        - [Normalized Error](#the-normalized-error)

        ***

        ## Try it here: Use the calculator

        ### [Newton-Raphson Calculator](#4977956d)

        - [The modified Formula](#1575ebfb)

        - Check it visually 👀: [Visualize your function](#visualize-the-function)
        """)
    
    st.subheader('Try some examples:')
    with st.expander('Examples from text book: "Métodos Numéricos para Ingenieros"\
                     - Steven C. Chapra, Raymond P. Canale'):
        with st.container(border=True):
            st.markdown("""
                    $$
                    2\\sin(\\sqrt{x}) - x 
                    $$
                    ```Python
                    2*sin(sqrt(x)) - x
                    ```
                    ---
                    $$
                    -1 + 5.5x  - 4x^{2} + 0.5x^{3}
                    $$
                    ```Python
                    -1 + 5.5*x - 4*x**2 + 0.5*x**3
                    ```
                    ---
                    $$
                    \\cos(x) - x\\sin({x})
                    $$
                    ```Python
                    cos(x) - x*sin(x)
                    ```
                    ---
                    $$
                    8\\sin({x})e^{-x} - 1
                    $$
                    ```Python
                    8*sin(x)*exp(-x) - 1
                    ```
                    """)


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

    The Newton-Raphson method is a powerful technique used to find
    approximations of the roots (solutions) of a real-valued function. 
    It's widely used in numerical analysis because of its simplicity and speed.

    The method is named after Sir Isaac Newton and Joseph Raphson. 
    Newton first introduced the concept in the 17th century, and 
    Raphson refined it later. Newton's contribution to calculus and 
    mathematical analysis laid the groundwork for many numerical methods, 
    including this one.

    ### Core Concepts

    **The goal** of the Newton-Raphson method is to find an approximate solution 
    to the equation $f(x) = 0$, where $f$ is a continuous and differentiable function.

    **The basic idea**: the method starts with an initial guess $x_0$ and iteratively 
    improves the guess using the function $f(x)$ and its derivative $f'(x)$.

    **The Newton-Raphson iteration formula is:**

    $$
    x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}
    $$

    Where:
    - $x_{n}$ is the current guess.
    - $x_{n+1}$ is the next (improved) guess.
    - $f(x_n)$ is the value of the function at $x_n$.
    - $f'(x_n)$ is the value of the derivative of the function at $x_n$.

    ### How It Works

    1. **Start with an initial guess** $x_0$.
    2. **Calculate the next guess** using the formula.
    3. **Repeat the process** until the guesses converge to a stable value (the root).

    **Let's see a basic example:**

    Imagine you want to find the root of the function $f(x) = x^2 - 2$.

    1. **Function and Derivative:**
    - $f(x) = x^2 - 2$
    - $f'(x) = 2x$

    2. **Initial Guess:**
    - Let's start with $x_0 = 1$.

    3. **Iteration Steps:**

    - **First Iteration:**
        $
        x_1 = x_0 - \\frac{f(x_0)}{f'(x_0)} = 1 - \\frac{1^2 - 2}{2 \\times 1} = 1 - \\frac{-1}{2} = 1.5
        $

    - **Second Iteration:**
        $
        x_2 = x_1 - \\frac{f(x_1)}{f'(x_1)} = 1.5 - \\frac{1.5^2 - 2}{2 \\times 1.5} = 1.5 - \\frac{0.25}{3} \\approx 1.4167
        $

    - **Third Iteration:**
        $
        x_3 = x_2 - \\frac{f(x_2)}{f'(x_2)} = 1.4167 - \\frac{1.4167^2 - 2}{2 \\times 1.4167} \\approx 1.4142
        $

    After a few iterations, you can see that the guesses are converging to $\\sqrt{2} \\approx 1.4142$.

    ### Advantages

    - **Fast Convergence:** When the initial guess is close to the actual root, the method converges very quickly.
    - **Simplicity:** The formula is straightforward and easy to implement.

    ### Disadvantages

    - **Initial Guess Sensitivity:** If the initial guess is far from the root, the method may fail to converge.
    - **Derivative Requirement:** The function must be differentiable, and computing the derivative can be complex for some functions.

    ### Conclusion

    The Newton-Raphson method is a powerful and widely-used technique for finding roots of functions.
    With its fast convergence and simple iterative approach, it is a valuable tool in numerical analysis 
    and various scientific computations.

    ---

    ## The Normalized Error

    In the context of the Newton-Raphson method, the normalized 
    error is a way to measure and evaluate the accuracy of the 
    iterative approximations of the root. By normalizing the error, we get a 
    relative measure that helps us understand how close the current approximation is to the actual root.

    The normalized error is defined as the absolute error of the current 
    approximation divided by the magnitude of the current approximation.
    Mathematically, it can be expressed as:

    $$
        \\text{Normalized Error} = \\left| \\frac{x_{n+1} - x_n}{x_{n+1}} \\right|
    $$

    where:
    - $x_n$ is the current approximation.
    - $x_{n+1}$ is the next (improved) approximation.

    ### Why it is important?

    The normalized error helps in:
    1. **Assessing Convergence:** It provides a way to assess how quickly the 
        iterative process is converging to the actual root.
    2. **Stopping Criterion:** It can be used as a stopping criterion for 
        the iterative process. When the normalized error falls below a predefined
        threshold, the iterations can be stopped, assuming that sufficient accuracy 
        has been achieved.

    So, **Keeping track of the normalized error** can halp you in your analysis. It is a useful metric 
    in the Newton-Raphson method for evaluating the accuracy and convergence of the iterative approximations. 
    It provides a relative measure of error that helps in making informed decisions about 
    stopping the iterations and assessing the performance of the method.

    """)

st.divider()


# --- "NORMAL METHOD" SECTION

st.markdown("""
            ### ✍️ Newton-Raphson Calculator:
            
            In this section you can easily use the Newton-Raphson method. This is a Python
            implementation of the formula, hosted here (thanks, Streamlit) to make it
            user-friendly. It can be either *iterative* or *recursive*. The main difference between
            these two approaches is the **stopping criterion**, where it can be a specific number
            of iterations or some precision threshold.

            Skipping all the mathematical details, this app perform the computations
            and display the **aproximations** and the **normalized errors** after each iteration.

            #### Type your function expression $f(x)$ """)

with st.container(border=True):
    
    col1, col2 = st.columns([0.5,0.5])

    with col1:            

        expr = st.text_input('Write your function here. Make sure you use **Python Syntax**', value='x',
                             help="Not sure how to write something? Look at some examples at the sidebar.")
        

        try:
        # Sympyfy the text input
            func = sympify(expr, rational=True).expand()

            # get latex representation
            func_latex = latex(func)

            st.latex(f"f(x) = {func_latex}")
        except:
            st.warning('Something went wrong. Make sure the expression has the correct syntax.')
        

    with col2.container(border=True):

        try:
            st.markdown("**First and Second Derivates:**")

            st.latex(f"f'(x) = {latex(func.diff(x).expand())}")

            st.latex(f"f''(x) = {latex(func.diff(x,x).expand())}")
        except:
            st.warning("Make sure you're using the correct Python Syntax. Try refreshing if the error continues.")

# Implementation of the method
st.markdown("""
            #### 📝 The Method implementation

            Choose how you want to address your specific problem, type your
            initial guess, adjust the other parametter and click on the **Run** 
            button to compute the root.
            """)
st.write("Newton-Raphson Formula: $x_{i+1} = x_{i} - \\frac{f(x_{i})}{f'(x_{i})}$")

try:
    # Instance the computer class
    NR = NewtonRaphson(function_expr=func, modified=False)
except:
    pass

@st.cache_data
def create_df(data):
    # create the dataframe with calculations info
    dataFrame = pd.DataFrame(
    data=[row[1:] for row in data],
    index=[row[0] for row in data],
    columns=['Root aproximation (x_i)', 'Normalized Error (%)']
    )
    return dataFrame

def save_to_session(data, val):
    # save dataframe and value to session state variables
    st.session_state['NR_data']['dataframe'] = data
    st.session_state['NR_data']['aprox'] = val

with st.container(border=True):
    param_col, data_col = st.columns([0.4,0.6]) 

    # Parametters column ---
    with param_col.container(border=True):

        approach = st.radio(
            'What approach do you want to use?',
            ['Recursion', 'Iteration'],
            captions=[
                'Choose the number of iterations',
                'Choose the precision for the *Normalized Error*'
            ]
        )

        st.write('⚙️ **Parametter adjusting**')
        init_val = st.number_input(
            'Initial guess $x_{0}$:',
            value=None,
            placeholder='int or float',
            key='NR_param init_val')
        

        if approach == 'Recursion':

            iters = st.number_input(
                'Number of iterations:',
                value=None,
                placeholder='Integer',
                format='%.0f',
            key='NR_param iters')

        elif approach == 'Iteration':

            prec = st.number_input(
                'Precision threshold (%):',
                value=None,
                placeholder='Float or integer',
                format='%.5f')

        
        # Button to call tha function
        run = st.button('Run', type='primary', key='NR call', on_click=NR_run)
        if run and approach == 'Recursion':
            # call the recursive method
            value = NR.compute_root(
                x_i=init_val,
                i=iters)
            save_to_session(NR.stored_aproximations, value)
            
        elif run and approach == 'Iteration':
            # Call the iterative method
            value = NR.compute_root_thr(
                x_i=init_val,
                precision=prec
            )
            save_to_session(NR.stored_aproximations, value)


    # Data column ---
    with data_col.container(border=True):        
        if st.session_state['NR_clicked']:

            try:
                df = create_df(st.session_state['NR_data']['dataframe'])

                # show the dataFrame
                st.dataframe(df, 
                        use_container_width=True, 
                        key='NR dataframe')
            except:
                st.error('Something went wrong when loading the iteration results. Try refreshing teh website.')

            # show the aprox value
            st.write(f"- **Computed root** $$x = {st.session_state['NR_data']['aprox']}$$"\
                     if type(st.session_state['NR_data']['aprox']) is not str else st.session_state['NR_data']['aprox'])
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
            'Initial guess $x_{0}$:',
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
    st.session_state.figure = False
    st.session_state.graph = None
    

params, graph = st.columns([0.2,0.8])

# Interval choosing column
with params:

    def interval_change():
        st.session_state.figure = False

    st.write('Choose the X-axis interval')
    x_lower = st.number_input('Lower bound', value=None, placeholder='x = ',
                              on_change=interval_change)
    x_upper = st.number_input('Upper bound', value=None, placeholder='x = ',
                              on_change=interval_change)

    plot_button = st.button('Show')

    if x_lower is None or x_upper is None:
        st.warning('Not providing values for lower or upper bounds may get an error.')

# Chart column
with graph.container(border=True):

    expression_latex = f"f(x) = {latex(func)}"
    # st.latex(expression_latex)
    st.latex(f"{expression_latex}, x \in I = [{x_lower if x_lower is not None else '-'}, \
                          {x_upper if x_upper is not None else '-'}]")

    if plot_button: 
        fig = text_book_chart(
                f=lambdify(x, func),
                interval=(x_lower,x_upper))
        st.session_state.figure = True
        st.session_state.graph = fig

    place_holder = st.empty()

    if st.session_state.figure == False:
        place_holder.caption(
            """Choose values for Lower and Upper bounds of the interval in which you want to visualize the function.
            Click 'Show' button and you'll see it. """
        )
    else:
        st.pyplot(st.session_state.graph)