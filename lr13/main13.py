# Есть: система обычных диф. ур-й (ОДУ)
#  
import numpy as np

def calc_k1(system, x, y_list, h) :
    return tuple(h * f(x, *y_list) for f in system)

def calc_k2(system, x, y_list, h, k1) :
    tmp_y = (y_list[i] + k1[i] * 0.5 for i in range(len(k1)))
    x = x + h * 0.5
    return calc_k1(system, x, tmp_y, h)

def calc_k3(system, x, y_list, h, k2) :
    return calc_k2(system, x, y_list, h, k2)

def calc_k4(system, x, y_list, h, k3) :
    tmp_y = (y_list[i] + k3[i] for i in range(len(k3)))
    x = x + h
    return calc_k1(system, x, tmp_y, h)

def calc_y(k1, k2, k3, k4, y_list) :
    new_k = ((k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(len(k1)))
    return (y_list[i] + new_k[i] for i in range(len(k1)))

def Runge_Kutta_methods(dif_equ_system, y0_list, a_b, h):
    a, b = a_b
    x_list = np.arange(a, b + h / 2, h)
    y_list = y0_list
    all_y = [y0_list]

    for x in x_list[ : -1] :
        k1 = calc_k1(dif_equ_system, x, y_list, h)
        k2 = calc_k2(dif_equ_system, x, y_list, h, k1)
        k3 = calc_k3(dif_equ_system, x, y_list, h, k2)
        k4 = calc_k4(dif_equ_system, x, y_list, h, k3)
        
        y_list = calc_y(k1, k2, k3, k4, y_list)

        all_y.append(y_list)
    
    return x_list[:-1], all_y, (k1, k2, k3, k4)

def Runge_Kutta_auto_h(dif_equ_system, y0_list, a_b, e) :
    calc_Egorov = lambda k1, k2, k3, k4 : 2 * (k1 - k2 - k3 + k4) / 3.0
    # a, b = a_b
    # h = (b - a)
    # x_list = np.arange(a, b + h / 2, h)
    # y_list = y0_list
    # all_y = [y0_list]

    a, b = a_b
    h = (b - a)
    x_list, all_y, k_list = Runge_Kutta_methods(dif_equ_system, y0_list, a_b, h)

    E = calc_Egorov(*k_list)
    if E <= (e / 2 ** 5) :
        h = h * 2
    elif E > e :
        h = h / 2
        ...
    elif E < e :
        ...





