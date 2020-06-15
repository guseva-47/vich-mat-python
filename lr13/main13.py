# Есть: система обычных диф. ур-й (ОДУ)
#  
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def calc_k1(system, x, y_list, h) :
    return [h * f(x, *y_list) for f in system]

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
    new_k = tuple((k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(len(k1)))
    return [y_list[i] + new_k[i] for i in range(len(k1))]

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
    
    return x_list, all_y, (k1, k2, k3, k4)

# def funk(dif_equ_system, y_list, x, h) :
    # k1 = calc_k1(dif_equ_system, x, y_list, h)
    # k2 = calc_k2(dif_equ_system, x, y_list, h, k1)
    # k3 = calc_k3(dif_equ_system, x, y_list, h, k2)
    # k4 = calc_k4(dif_equ_system, x, y_list, h, k3)
    
    # y_list = calc_y(k1, k2, k3, k4, y_list)

    # return y_list


def Runge_Kutta_auto_h(dif_equ_system, y0_list, a_b, e, MIN_H) :
        
    a, b = a_b
    new_a = a
    h = (b - a) / 2
    
    x_list = [a]
    all_y = [y0_list]

    MIN_E = e / 2**5
    
    while new_a < b :
    
        # Нахождение y2 с шагом 2h
        a_b = (new_a, new_a + 2 * h)
        _, all_y_2h, _ = Runge_Kutta_methods(dif_equ_system, y0_list, a_b, 2 * h)
        
        # if len(all_y_2h) < 2 : break
        assert len(all_y_2h) > 1

        y_2h = all_y_2h[-1]

        while True :
            # Нахождение y2 с шагом h. 
            a_b = (new_a, new_a + 2 * h)
            _, all_y_h, _ = Runge_Kutta_methods(dif_equ_system, y0_list, a_b, h)

            assert len(all_y_h) > 2

            y_h = all_y_h[-1]
        
            # абсолютная погрешность TODO
            tmp = (abs((y_h[i] - y_2h[i])/( 2**4 - 1)) for i in range(len(y_h)))
            err = np.fromiter(tmp, float)
            err = np.linalg.norm(err)

            if err > e :
                h = h / 2.0
                y_2h = all_y_h[-2]
            # Если при некотором шаге ℎ выполняется неравенство 𝑒𝑟𝑟 ≤ 𝜀, то считается, что значения 𝑦1, 𝑦2 решения в точках 𝑥0 + ℎ, 𝑥0 + 2ℎ найдены верно
            elif err <= e or h < MIN_H :
                # запись результата
                all_y.append(all_y_h[-2])
                all_y.append(all_y_h[-1])

                x_list.append(new_a + h)
                x_list.append(new_a + 2 * h)

                # возможно err чересчур мала
                h = h if err >= MIN_E else 2 * h

                new_a += 2 * h

def nested_auto_step(dif_equ_system, y0_list, a_b, e, MIN_H) :
    def calc_local_error(k1, k2, k3, k4):
        calc_Egorov = lambda k1, k2, k3, k4 : abs(2 * (k1 - k2 - k3 + k4) / 3.0)
        return max([calc_Egorov(k1[i], k2[i], k3[i], k4[i]) for i in range(len(k1))])

    a, b = a_b
    new_a = a
    h = (b - a)
    
    x_list = [a]
    all_y = [y0_list]

    MIN_E = e / 2**5
    
    while new_a < b :

        if new_a + h > b : h = b - h

        while True :
            y_list = all_y[-1]
            a_b = (new_a, new_a + h)
            _, all_y_n, all_kn = Runge_Kutta_methods(dif_equ_system, y_list, a_b, h)

            err = calc_local_error(*all_kn)

            if err > e :
                h = h / 2.0
            # Если при некотором шаге ℎ выполняется неравенство 𝑒𝑟𝑟 ≤ 𝜀, то считается, что значения 𝑦1, 𝑦2 решения в точках 𝑥0 + ℎ, 𝑥0 + 2ℎ найдены верно
            elif err <= e or h < MIN_H :
                # запись результата
                all_y.append(all_y_n[-1])
                x_list.append(new_a + h)

                # возможно err чересчур мала, тогда текущий результат принимается, но следующий шаг будет больше
                h = h if err >= MIN_E else 2 * h
                new_a += h
                break

    return x_list, all_y

# def calc_new_h(f_system, x, right, cur_h, prev_y, precision):
#     h = cur_h
#     if x + h > right:
#         return right - x
#     while True:
#         _k1 = calc_k1(h, f_system, x, prev_y)
#         _k2 = calc_k2(h, _k1, f_system, x, prev_y)
#         _k3 = calc_k3(h, _k2, f_system, x, prev_y)
#         _k4 = calc_k4(h, _k3, f_system, x, prev_y)
#         error = calc_local_error(_k1, _k2, _k3, _k4)
#         if error > precision:
#             h = h / 2
#             # h = h * (precision / max(errors))**(1 / 5) * 0.8
#         elif error < precision / 32:
#             h = h * 2
#             break
#         if x + h > right:
#             return right - x
#         # h = h * (precision / max(errors))**(1 / 5) * 0.8
#         else:
#             break
#     return h

# def calc_local_error(_k1, _k2, _k3, _k4):
#     return max([abs(2 * (_k1[i] - _k2[i] - _k3[i] + _k4[i]) / 3) for i in range(len(_k1))])

# def solve_system_with_auto_step(left, right, precision, f_system, prev_y):
#     h = (right - left)
#     cur_x = left
#     x_array = [cur_x]
#     y_array = [prev_y]
#     while cur_x < right:
#         h = calc_new_h(f_system, cur_x, right, h, prev_y, precision)
#         _k1 = calc_k1(h, f_system, cur_x, prev_y)
#         _k2 = calc_k2(h, _k1, f_system, cur_x, prev_y)
#         _k3 = calc_k3(h, _k2, f_system, cur_x, prev_y)
#         _k4 = calc_k4(h, _k3, f_system, cur_x, prev_y)
#         local_errors.append(calc_local_error(_k1, _k2, _k3, _k4))
#         next_y = calc_next_y(_k1, _k2, _k3, _k4, prev_y)
#         cur_x = cur_x + h
#         x_array.append(cur_x)
#         y_array.append(next_y)
#         prev_y = next_y
#     return x_array, y_array

def to_drow(x_list, y_all, a_b, exact_f_s) :
    fig, ax = plt.subplots()
    for i in range(len(y_all[0])) :
        ax.plot(x_list, tuple(y[i] for y in y_all))
    
    ex_x_list = np.arange(*a_b, 0.001)
    for f in exact_f_s :
        ax.plot(ex_x_list, tuple(map(f, ex_x_list)), label='reference')
    plt.show()


a, b = -2, 2
f_system = [ lambda x, y: 2 * x + y - x**2 ]
exact_f_system = [ lambda x: x**2 ]
start_y = [f(a) for f in exact_f_system]

x, y, _ = Runge_Kutta_methods(f_system, start_y, (a, b), 0.6)
x2, y2 = nested_auto_step(f_system, start_y, (a, b), 0.1, 0.0000001)

to_drow(x, y, (a,b), exact_f_system)
to_drow(x2, y2, (a,b), exact_f_system)
