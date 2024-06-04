from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

x = Symbol('x')
f, df, df2 = symbols('f, df, df2', cls=Function)

stored_values = []

# The system
def Newton_Raphson(x_i, i, **kwargs):
    """## Newton Raphson Method
    Using the formula for Newton-Rapshon recursion to compute 
    the roots of a real-valued continuous and differentiable 
    function, given an initial value.

    ### Parametters
    - func_expr: The function's expression. A function of the independent variable ' x '.

    - x_i: Short for 'x initial'. The initial value for the recursion
    to start with.

    - i: The number of iterations desired.

    - *args: When being used in third party programs, pass the functions and its derivates as 
    additional arguments (f, df, *df2)
    """
    # Function passed as key arg or default x
    func = kwargs.get('sympy_expr', x)

    # Converted to a python function
    f = lambdify(x, func)
    # compute the first derivate and convert to a python function
    df = lambdify(x, func.diff(x))

    # make a list to store values
    global stored_values

    if i == 0:
        # print('Initial value: x_0 =', x_i)
        return x_i
    else:
        # Reduce the iteration number progresively
        i -= 1
        
        # Compute for the immediately previous value
        x_I = Newton_Raphson(x_i, i, **kwargs)

        # NR formula 
        x_II = x_I - (f(x_I) / df(x_I))

        # relative error
        rel_error = (abs(x_II - x_I) / abs(x_II))

        stored_values.append((i+1, x_II, rel_error))

        # print(f'Iteration {i+1}  |  x_{i+1} = {x_II}, relative error: {round(rel_error, 6)}')
        return x_II
    
# Modified method
def Newton_Raphson_modified(x_i, i, **kwargs):
    """## Newton Raphson Method
    Using the formula for Newton-Rapshon recursion to compute 
    the roots of a real-valued continuous and differentiable 
    function, given an initial value.

    ### Parametters
    - func_expr: The function's expression. A function of the independent variable ' x '.

    - x_i: Short for 'x initial'. The initial value for the recursion
    to start with.

    - i: The number of iterations desired.
    """

    # Function passed as key arg or default x
    func = kwargs.get('sympy_expr', x)

    # Converted to a python function
    f = lambdify(x, func)
    df = lambdify(x, func.diff(x))
    df2 = lambdify(x, func.diff(x,2))

    if i == 0:
        # print('Initial value: x_0 =', x_i)
        return x_i
    else:
        # Reduce the iteration number progresively
        i -= 1
        
        # Compute for the immediately previous value
        x_I = Newton_Raphson_modified(x_i, i, **kwargs)

        # NR formula 
        x_II = x_I - (f(x_I)*df(x_I)) / (df(x_I)**2 - f(x_I)*df2(x_I))

        # relative error
        rel_error = (abs(x_II - x_I) / abs(x_II))

        # print(f'Iteration {i+1}  |  x_{i+1} = {x_II}, relative error: {round(rel_error, 6)}')
        return x_II
    

# Plotting function
def text_book_chart(func:Function, interval:tuple = (-10, 10)):
    
    lower = interval[0]
    upper = interval[1]

    x = np.linspace(lower, upper, 100)

    # Plot
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(x,func(x)) # It also works with sympy function f(x)

    # To get a "text book" look: relocate the spines of the figure
    ax.spines[["left", "bottom"]].set_position('zero')
    ax.spines[["top", "right"]].set_visible(False)

    ax.set_xticks(np.arange(lower, upper+2, 2)) # modify the xticks

    # setting the name of x and y axis
    ax.set_xlabel('x', loc='right', fontstyle='italic', fontsize='large')
    ax.set_ylabel('f(x)', loc='top', fontstyle='italic', fontsize='large', rotation='horizontal')
    ax.grid(True, which='both') # plotting a grid
    
    # plt.show()
    return fig
    

if __name__ == '__main__':

    x, y, z = symbols('x y z')

    # Ask for the function expression
    expr = input('Write your function (acordding to python syntax): ')

    # convert it to sympy expression object and print
    func = sympify(expr).expand()
    print(f"f(x) = {func} \nf'(x) = {func.diff(x)}")

    # Ask for the parametters
    init_val = float(input('Type the initial value X_0 =  '))
    iters = int(input('How many iterations of the method? Enter an integer value: '))

    # execute the function
    val = Newton_Raphson(
        x_i=init_val,
        i=iters,
        sympy_expr=func)
    
    print('aproximation', val)
    print(len(stored_values))

    df = pd.DataFrame(
        data=[row[1:] for row in stored_values],
        index=[f'i = {int(row[0])}' for row in stored_values],
        columns=['x_i', 'Normalized Error']
        )
    print(df)
