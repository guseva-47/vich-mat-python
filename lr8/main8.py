import random as rand
import math
import numpy as np
import matplotlib.pyplot as plt

# Таблица значений
def calc_table(funk, start, stop, step) :

    x_list = list(np.arange(start, stop + step / 2, step))

    table = tuple(map(lambda x: (x, funk(x)), x_list))
    return table

# Первая производная
def first_derivative(table, step) :
    result = []
    const = 1 / (2 * step)
    for i in range(1, len(table) - 1) :
        y0 = table[i - 1][1]
        y2 = table[i + 1][1]
        result.append((table[i][0], (y2 - y0) * const))
    
    return result

# Вторая производная
def second_derivative(table, step) :
    result = []
    const = 1 / (step**2)
    for i in range(1, len(table) - 1) :
        
        y0 = table[i - 1][1]
        y1 = table[i][1]
        y2 = table[i + 1][1]
        result.append((table[i][0], (y2 + y0 - 2 * y1) * const))
    
    return result

# Третья производная
def thrid_derivative(table, step) :
    
    result = []
    const = 1 / (2 * step**3)
    for i in range(2, len(table) - 2) :
                
        y0 = table[i - 2][1]
        y1 = table[i - 1][1]
        y3 = table[i + 1][1]
        y4 = table[i + 2][1]
        new_y = (y4 - 2 * y3 + 2 * y1 - y0) * const
        result.append((table[i][0], new_y))
    
    return result

# list1, list2 -- списки в которых по три массива в каждом, значения для первой, второй и третьей производных
def calc_error(deriv, reference_deriv) :
    errors = []
    sdvig = 0
    for i in range(len(reference_deriv)) :
        # в производной теряются значения по краям исходной функции, укоротим исходную функцию
        t_deriv, t_ref = deriv[i], reference_deriv[i]
        sdvig = abs(len(t_deriv) - len(t_ref)) // 2
        t_ref = t_ref[sdvig : -sdvig] if sdvig > 0 else t_ref

        funk = lambda n1, n2 : (n1[0], abs(n1[1] - n2[1]))
        tmp = tuple(funk(t_deriv[j], t_ref[j]) for j in range(len(t_ref)))
        errors.append(tmp)
        
    return errors

def draw(referense_list, der_list, func_title, start, stop, step, errors_list, func_error_title):
    _, calls = plt.subplots(2, 3)
    for i in range(3) :
        ax = calls[0][i]
        ax.set_title(func_title[i])

        x, y = tuple(map(lambda n: n[0], referense_list[i])), tuple(map(lambda n: n[1], referense_list[i]))
        ax.plot(x, y, linewidth = 3)

        x, y = tuple(map(lambda n: n[0], der_list[i])), tuple(map(lambda n: n[1], der_list[i]))
        ax.plot(x, y)

    for i in range(3) :
        ax = calls[1][i]
        max_error = max(map(lambda n: n[1], errors_list[i]))
        title = f'{func_error_title[i]} max ошибка = {max_error : .5f}'
        ax.set_title(title)

        x, y = tuple(map(lambda n: n[0], errors_list[i])), tuple(map(lambda n: n[1], errors_list[i]))
        ax.plot(x, y)

    plt.show()

def full_cycle(funk, referense_derivs_funks, max_e, START, STOP, STEP, e_str, funk_alias) :
    table = calc_table(funk, START, STOP, STEP)

    derivativs = []
    derivativs.append(first_derivative(table, STEP))
    derivativs.append(second_derivative(table, STEP))
    derivativs.append(thrid_derivative(table, STEP))

    referense_derivs = []
    for funk in referense_derivs_funks :
        referense_derivs.append(calc_table(funk, START, STOP, STEP))

    errors = calc_error(derivativs, referense_derivs)

    draw(referense_derivs, derivativs, [funk_alias + '\n\nпервая производная', 'шаг = '+ str(STEP) + '\n\nвторая', 'третья'], 
    START, STOP, STEP, errors, ['возмущения <= ' + e_str+', ', '', ''])


def main() :
    START = -1
    STOP = 1
    funk_alias = 'y = sin(2x)'
    

    referense_derivs_funks = (
        lambda x : 2 * math.cos(2 * x),
        lambda x: -4 * math.sin(2 * x),
        lambda x: -8 * math.cos(2 * x)
    )

    max_e = (0, 0.001, 0.000001)
    for step in [0.01, 0.001, 0.0001] :
        for e in max_e :
            funk = lambda x: math.sin(2 * x) + rand.uniform(-e, e)
            full_cycle(funk, referense_derivs_funks, max_e, START, STOP, step, str(e), funk_alias)


if __name__ == "__main__":
    main()    