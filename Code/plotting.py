import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os.path

def set_plot_params(fig_width=1, fig_height=1):
    MM_TO_INCH = 1/25.4
    mpl.rc("text", usetex=False)
    plt.rc("figure", autolayout=True)
    plt.rc('lines', linewidth=1, color='r')
    plt.rc("legend", fontsize=6)
    plt.rc("font", family="serif", size=8)
    plt.figure(figsize=(85*fig_width*MM_TO_INCH, 70*fig_height*MM_TO_INCH))
    plt.ticklabel_format(axis = "y", style="sci")

if __name__ == "__main__":
    filename = "num_its"

    set_plot_params(1,1)

    x = np.round(np.linspace(2, 200, 20))
    Ns8 = np.array([1,208, 901, 1869, 3341, 5122, 7439, 9886, 12729, 16245, 19852, 24290, 28681, 33339, 38888, 44326, 50696, 57128, 64236, 70691])
    Ns7 = np.array([1,202, 874, 1796, 3222, 4956, 7187, 9553, 12307, 15702, 19164, 23525, 27811, 32252, 37604, 42901, 49007, 55281, 62156, 68467])
    Ns8Dense = np.array([1, 210, 842, 1808, 3267, 4935, 7268, 9714, 12616, 16117, 19788, 24147, 28672, 33280, 39167, 44411, 51386, 57207, 65015, 71541])

    x2 = x**2
    two_x2 = 2*x**2
    three_x2 = 3*x**2
    x3 = x**3
    
    #plt.plot(x, Ns7, linestyle="-", marker="o", label = r"Jacobi's, $\varepsilon = 10^{-7}$")
    plt.plot(x,two_x2, label = r"$2N^2$")
    #plt.plot(x, Ns8Dense, linestyle="-", marker="o", label=r"Dense Jacobi's $\varepsilon = 10^{-8}$")
    plt.plot(x,x2, label=r"$N^2$")
    plt.plot(x, Ns8, linestyle="-", marker=".", label = r"Jacobi's, $\varepsilon = 10^{-8}$")
    #plt.plot(x,three_x2, label=r"$3x^2$")
    #plt.plot(x, x3, label = r"$x^3$")
    plt.xlabel(r"$N$")
    plt.ylabel("Iterations")
    plt.legend()
    plt.show()


