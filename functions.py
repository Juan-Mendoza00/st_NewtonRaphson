from sympy import *
import numpy as np
import matplotlib.pyplot as plt
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

    - x_i: Short for 'x initial'. The initial value for the recursion
    to start with.

    - i: The number of iterations desired.

    Pass functions as arguments
    - *kwargs: (f, df, df2) Pass the lamdified function, lambdified first derivate
    and lambdified second derivate.
    """
    # default function
    func = x**3 + x**2 + x

    # Convert or get passed functions
    f = kwargs.get('f', lambdify(x, func))
    df = kwargs.get('df',lambdify(x, func.diff(x)))

    # scope to the list to store values
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
mdf_stored_values = []
def Newton_Raphson_modified(x_i, i, **kwargs):
    """## Newton Raphson Method
    Using the formula for Newton-Rapshon recursion to compute 
    the roots of a real-valued continuous and differentiable 
    function, given an initial value.

    ### Parametters

    - x_i: Short for 'x initial'. The initial value for the recursion
    to start with.

    - i: The number of iterations desired.
    
    Pass the function and its derivates
    - f, df, df2: Lambdified functions to make numerical calculations.
    """

    # Function passed as key arg or default x
    func = x**3 + x**2 + x

    # Converted to a python function
    f = kwargs.get('f', lambdify(x, func))
    df = kwargs.get('df',lambdify(x, func.diff(x)))
    df2 = kwargs.get('df2',lambdify(x, func.diff(x,2)))

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

        global mdf_stored_values
        mdf_stored_values.append((i+1, x_II, rel_error))

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
    

def main():
    x, y, z = symbols('x y z')

    # Ask for the function expression
    expr = input('Write your function (acordding to python syntax): ')

    # convert it to sympy expression object and print
    func = sympify(expr).expand()
    print(f"f(x) = {func} \nf'(x) = {func.diff(x)}\nf''(x) = {func.diff(x,2)}")

    # Ask for the parametters
    init_val = float(input('Type the initial value X_0 =  '))
    iters = int(input('How many iterations of the method? Enter an integer value: '))

    # lambdify
    f = lambdify(x, func)
    df = lambdify(x, func.diff(x))
    df2 = lambdify(x, func.diff(x,2))

    # execute the function
    val = Newton_Raphson(
        x_i=init_val,
        i=iters,
        f=f,
        df=df)

    # Dataframe with iteration info
    df = pd.DataFrame(
        data=[row[1:] for row in stored_values],
        index=[f'i = {int(row[0])}' for row in stored_values],
        columns=['x_i', 'Normalized Error']
        )
    print(df)

    print('aproximation', val)

if __name__ == '__main__':
    main()
