from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
This module contains the Class objecto corresponding to Newton-Raphson
method for Numerical Analysis
"""

x = Symbol('x')
f = Function('f')

class NewtonRaphson:
    """Creates an instance to compute roots of a function using the Newton-Raphson's
    method on the expression passed as the function argument."""

    def __init__(self,
                 function_expr,
                 modified:bool = False) -> None:
        
        self.function = function_expr
        self.modified = modified
        self.stored_aproximations = []
        # convert function to python functions to perform calculations
        try:
            self.f_x = lambdify(x, function_expr)
            self.df_dx = lambdify(x, function_expr.diff(x))
        except:
            print('Incorrect syntax for the function expression')
            raise SyntaxError
        
        if modified:
            self.df2_dx2 = lambdify(x, function_expr.diff(x,x))
        pass

    def __str__(self):
        if self.modified:
            return "Modified Newton-Raphson computer"
        return "Newton-Raphson computer instance"

    # Method implementation
    def compute_root(self, x_i, i:50, prec=None):
        
        """## Newton Raphson Method
        Using the formula for Newton-Rapshon recursion to compute 
        the roots of a real-valued continuous and differentiable 
        function, given an initial value.

        ### Parametters

        x_i (float): 
            Short for 'x initial'. The initial value for the recursion
            to start with.

        i (int): 
            The number of iterations desired.
        
        Optional: Choose a precision threshold as stoppage criteria for the method using prec parametter.

        prec (float) Default None:
            The method will stop computing aproximations when the Normalized Error after each iteration
            is iqual or less than :prec.

        """

        # Verify if the initial value is already a root
        val = self.f_x(x_i)
        dist_to_zero = abs(0-val)
        if dist_to_zero < 0.00005:
            return f"x = {x_i} already aproximates a root." 

        if i==0:

            # Every time the iteration counter reaches zero it will clean
            # the list of stored aproximations. That avoids that list 
            # to grow infinitely.
            self.stored_aproximations.clear()

            # Once is cleaned, appends the first row and start over again.
            self.stored_aproximations.append((f"i={i}", x_i, '--'))

            return x_i
        
        else:
            # Reduce the iteration number progresively
            i -= 1

            # Compute for the immediately previous value
            x_I = self.compute_root(x_i, i, prec)

            if self.modified == True:
                # Modified NR formula 
                x_II = x_I - (self.f_x(x_I)*self.df_dx(x_I)) / (self.df_dx(x_I)**2 - self.f_x(x_I)*self.df2_dx2(x_I))
            else:
                # Simple NR formula 
                x_II = x_I - (self.f_x(x_I) / self.df_dx(x_I))

            # Normalized error
            rel_error = str((abs(x_II - x_I) / abs(x_II)) * 100) + ' %'

            # store every iteration value        
            self.stored_aproximations.append((f"i={i+1}", x_II, rel_error))
            
            return x_II
    
    # Iterative function ---
    def compute_root_thr(self, x_i:float, precision:float):

        norm_error = 100

        root_aprox = None
        prev_x = x_i

        i = 0 

        self.stored_aproximations.clear()
        self.stored_aproximations.append((f"i={i}", x_i, '--'))

        while norm_error > precision:
            
            i += 1

            if self.modified == True:
                # Modified NR formula 
                root_aprox = prev_x - (self.f_x(prev_x)*self.df_dx(prev_x)) / (self.df_dx(prev_x)**2 - self.f_x(prev_x)*self.df2_dx2(prev_x))
            else:
                # Simple NR formula 
                root_aprox = prev_x - (self.f_x(prev_x) / self.df_dx(prev_x))

            norm_error = abs(root_aprox - prev_x) / abs(root_aprox) * 100 # % (percentage)

            self.stored_aproximations.append((f"i={i}", root_aprox, norm_error))

            # Update previous x variable
            prev_x = root_aprox

            if i == 20:
                break
            
        return root_aprox

    
# Plotting function
def text_book_chart(f:Function, interval:tuple = (-10, 10)):
    
    lower = interval[0]
    upper = interval[1]

    interval_lenght = abs(upper-lower)

    x = np.linspace(lower, upper, 500)
    y = f(x)

    # Plot
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(x,y)

    # To get a "text book" look: relocate the spines of the figure
    if 0 not in range(int(lower),int(upper) + 1):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_position('zero')
    else:
        ax.spines[["left", "bottom"]].set_position('zero')
        ax.spines[["top", "right"]].set_visible(False)

    # Customize xticks
    if interval_lenght < 8:
        xticks = np.linspace(lower, upper, 11)
    else:
        xticks = np.arange(lower, upper+1, 1)
        
    ax.set_xticks(xticks) # modify the xticks

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
    # iters = int(input('How many iterations of the method? Enter an integer value: '))
    precision = float(input('Choose precision (%) for normalized error'))
    modified = bool((input('Modified Newton-Raphson? [true/false]: ')).capitalize())

    NR = NewtonRaphson(function_expr=func, modified=modified)
    
    # val = NR.compute_root(
    #     x_i=init_val, # initial x_
    #     i=iters,    # Number of iterations
    #     )

    val = NR.compute_root_thr(x_i=init_val,
                              precision=precision)

    # Dataframe with iteration info
    df = pd.DataFrame(
        data=[row[1:] for row in NR.stored_aproximations],
        index=[row[0] for row in NR.stored_aproximations],
        columns=['x_i', 'Normalized Error (%)']
        )
    print(df)

    print('aproximation', val)

if __name__ == '__main__':
    main()