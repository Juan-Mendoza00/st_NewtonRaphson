from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = Symbol('x')

stored_values = []

# Method implementation
def Newton_Raphson(
        func,
        x_i:float,
        i:int,
        **kwargs):
    
    """## Newton Raphson Method
    Using the formula for Newton-Rapshon recursion to compute 
    the roots of a real-valued continuous and differentiable 
    function, given an initial value.

    ### Parametters

    func (sympy.core... expression):
        A real-valued function expression.

    x_i (float): 
        Short for 'x initial'. The initial value for the recursion
        to start with.

    i (int): 
        The number of iterations desired.

    --- Pass functions as arguments\n
    *kwargs.(f, df, df2) - Optional: 
        Pass the lamdified function, lambdified first derivate
        and lambdified second derivate.

    --- Choose between simple or modified method\n
    *kwargs.mod (bool) - Optional:
        Default False. If true, it will compute the aproximation using the modified
        method.
    """

    # Convert -func- or get passed functions (optional)
    # Note: derivates can be passed instead of being computed below
    f = kwargs.get('f', lambdify(x, func))
    df = kwargs.get('df',lambdify(x, func.diff(x)))
    df2 = kwargs.get('df2',lambdify(x, func.diff(x,2)))

    mod = kwargs.get('mod', False)

    # To store the aproximations
    stored_values = []
    if i == 0:
        stored_values.append((f"i={i}", x_i, '--'))
        return x_i
    else:
        # Reduce the iteration number progresively
        i -= 1
        
        # Compute for the immediately previous value
        x_I = Newton_Raphson(x_i, i, **kwargs)

        if mod:
             # Modified NR formula 
            x_II = x_I - (f(x_I)*df(x_I)) / (df(x_I)**2 - f(x_I)*df2(x_I))
        else:
            # Simple NR formula 
            x_II = x_I - (f(x_I) / df(x_I))

        # Normalized error
        rel_error = str((abs(x_II - x_I) / abs(x_II)) * 100) + ' %'

        # store iteration values        
        stored_values.append((f"i={i+1}", x_II, rel_error))

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
    x= symbols('x')

    # Ask for the function expression
    expr = input('Write your function (acordding to python syntax): ')

    # convert it to sympy expression object and print
    func = sympify(expr).expand()
    print(f"f(x) = {func} \nf'(x) = {func.diff(x)}\nf''(x) = {func.diff(x,2)}")

    # Ask for the parametters
    init_val = float(input('Type the initial value X_0 =  '))
    iters = int(input('How many iterations of the method? Enter an integer value: '))
    mod = (input('Modified Newton-Raphson? [true/false]: ')).capitalize()

    # lambdify
    f = lambdify(x, func)
    df = lambdify(x, func.diff(x))
    df2 = lambdify(x, func.diff(x,2))

    # execute the function
    val = Newton_Raphson(
        x_i=init_val, # initial x_
        i=iters,    # Number of iterations
        f=f,
        df=df,
        df2=df2,
        mod=mod)

    # Dataframe with iteration info
    df = pd.DataFrame(
        data=[row[1:] for row in stored_values],
        index=[row[0] for row in stored_values],
        columns=['x_i', 'Normalized Error (%)']
        )
    print(df)

    print('aproximation', val)

if __name__ == '__main__':
    main()
