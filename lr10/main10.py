import math
import numpy as np
import random as rand
import matplotlib.pyplot as plt

def find_index(table, x, begin, end):
    if begin >= end:
        raise ValueError('x выходит за пределы таблицы')
    
    mid = (begin+end)//2
    
    if x < table[mid][0]:
        return find_index(table, x, begin, mid)
    if x > table[mid+1][0]:
        return find_index(table, x, mid+1, end)
    else:
        return mid

# Возвращает значение кубического сплайна дефекта 1 в точке x
def spline(x, table, spline_values):
    i = find_index(table, x, 0, len(table)-1)+1
    x_i = table[i][0]
    
    A,B,C,D = spline_values
    
    return A[i] + B[i]*(x-x_i) + C[i]/2*(x-x_i)**2 + D[i]/6*(x-x_i)**3

def calc_spline_values_for_nodes(table):
    '''Считает для сплайна коэффициенты a,b,c,d из таблицы значений функции'''

    # большие буквы - списки, маленькие - элементы списков
    Y = [y for _, y in table]
    H = [None]+[next_x - x for (next_x, _), (x, _) in zip(table[1:], table[:-1])]

    # прямой ход метода прогонки
    alpha = [None]+[0]*(len(Y))
    beta = [None]+[0]*len(alpha)

    for i in range(1, len(Y)-1) :
        alpha[i+1] = H[i+1]/(-2*(H[i]+H[i+1]) - H[i]*alpha[i])

        up = H[i]*beta[i] - 6*((Y[i+1]-Y[i])/H[i+1] - (Y[i]-Y[i-1])/H[i])
        down = -2*(H[i]+H[i+1]) - H[i]*alpha[i]
        beta[i+1] = up/down

    # обратный ход метода прогонки
    C = [0]*len(Y)
    for i in range(len(C)-2, 0, -1) :
        C[i] = alpha[i+1]*C[i+1] + beta[i+1]

    # вычисление остальных коэффициентов
    D = [None]+[(next_c - c) / h for next_c, c, h in zip(C[1:], C[:-1], H[1:])]
    B = [None]+[h/2*c - h*h/6*d + (next_y-y)/h for c, d, h, next_y, y in zip(C[1:], D[1:], H[1:], Y[1:], Y[:-1])]

    return Y, B, C, D

# Первая производная
def first_derivative(table, step) :
    result = []
    const = 1 / (2 * step)
    for i in range(1, len(table) - 1) :
        y0 = table[i - 1][1]
        y2 = table[i + 1][1]
        result.append((table[i][0], (y2 - y0) * const))
    
    return result

# генерирует исходную таблицу значений кривой
def gen_table():
    x_func = lambda t: math.sin(t)
    y_func = lambda t: math.cos(t)

    t_list = np.linspace(-0.5 * math.pi, 0.5 *math.pi, 8)

    return tuple((x_func(t), y_func(t)) for t in t_list)

# # генерирует подинтегральную функцию
# # возвращает границы интервала для t и функцию от t
# def under_integral_func(table) :
#     x_table = tuple((i, x) for i, (x, _) in enumerate(table))
#     y_table = tuple((i, y) for i, (_, y) in enumerate(table))[1:-1]

#     dx_table = first_derivative(x_table, abs(table[0][0]-table[0][1]))

#     dx_spline_values = calc_spline_values_for_nodes(dx_table)
#     y_spline_values = calc_spline_values_for_nodes(y_table)

#     dx_spline_func = lambda t: spline(t, dx_table, dx_spline_values)
#     y_spline_func = lambda t: spline(t, y_table, y_spline_values)

#     res_func = lambda t: abs(y_spline_func(t) * dx_spline_func(t))
#     a, b = y_table[0][0], y_table[-1][0]
#     return a, b, res_func

def under_integral_func(table) :
    x_table = tuple((i, x) for i, (x, _) in enumerate(table))[1:-1]
    y_table = tuple((i, y) for i, (_, y) in enumerate(table))

    dy_table = first_derivative(y_table, abs(x_table[0][0] - x_table[1][0]))

    x_spline_values = calc_spline_values_for_nodes(x_table)
    dy_spline_values = calc_spline_values_for_nodes(dy_table)

    dy_spline_func = lambda t: spline(t, dy_table, dy_spline_values)
    x_spline_func = lambda t: spline(t, x_table, x_spline_values)

    res_func = lambda t: abs(dy_spline_func(t) * x_spline_func(t))
    a, b = dy_table[0][0], dy_table[-1][0]
    return a, b, res_func

# Метод Симпсона
def Simpson_method(funk, a, b, h) :
    even = tuple(np.arange(a + 2 * h, b, 2 * h))
    odd = tuple(np.arange(a + h, b, 2 * h))
    a_b = tuple(np.arange(a, b + h/2, h))
    a_b = (a_b[0], a_b[-1])

    even = tuple(map(funk, even))
    odd = tuple(map(funk, odd))
    a_b = tuple(map(funk, a_b))

    result = 2 * sum(even) + 4 * sum(odd) + sum(a_b)
    result = result * h / 3
    return result

# Автовыбор шага для любого метода
def auto_simpson(funk, a, b, e) :
    last_result = None
    new_result = None
    h = abs(a - b) / 2
    # ограничение на число разбиений
    for count in range(10000) :
        new_result = Simpson_method(funk, a, b, h)
        if last_result and abs(new_result - last_result) < e :
            break
        h /= 2
        last_result = new_result
    return new_result, count

def Monte_Carlo_method(dots_in_area, dots_out_area, s_rectangle) :
    
    return s_rectangle * len(dots_in_area) / (len(dots_in_area) + len(dots_out_area))



# table -- область, у каторой ищется граница. нарисовать полученную область
table = gen_table()

# a -- начало отрезка, b -- конец отрезка, функция y(t)
a, b, funk = under_integral_func(table)

# вычисление площади с помощью метода симпсона
s_simpson, _ = auto_simpson(funk, a, b, 0.00001)


def fu(x_list, y_list, count) :
    # вычисление площади с помощью метода Монте-Карло
        # ограничить область четырехгранником
        # нагенерировать в области n точек
        # определить какие точки входят в фигуру, а какие нет 
    x_max, x_min = max(x_list), min(x_list)
    y_max, y_min = max(y_list), min(y_list)

    def is_in_area(x_list, y_list, dot) :
        answr = False
        n = len(x_list)
        for i, j in zip(range(0, n), range(-1, n-1)) :
            x_i, y_i = x_list[i], y_list[i]
            x_j, y_j = x_list[j], y_list[j]
            x, y = dot
            if (y_i <= y < y_j or y_j <= y < y_i) and (y_j - y_i) != 0 :
                x_intersection = x_i + (x_j - x_i) * (y - y_i) / (y_j - y_i)
                if x > x_intersection : 
                    answr = not answr
        return answr

    new_dots = tuple((rand.uniform(x_min, x_max), rand.uniform(y_min, y_max)) for _ in range(count))

    is_in_area_list = map(lambda dot: is_in_area(x_list, y_list, dot), new_dots)

    dots_in, dots_out = [], []
    for i, ok in enumerate(is_in_area_list) :
        if ok :
            dots_in.append(new_dots[i])
        else :
            dots_out.append(new_dots[i])

    # dots_in = tuple(filter(lambda n: n, is_in_area_list))
    # dots_out = tuple(filter(lambda n: not n, is_in_area_list))

    return dots_in, dots_out, (x_min, x_max), (y_min, y_max)



# DROW

fig, ax = plt.subplots(1, 1)

x_table = tuple((i, x) for i, (x, _) in enumerate(table))
y_table = tuple((i, y) for i, (_, y) in enumerate(table))

x_spline_values = calc_spline_values_for_nodes(x_table)
y_spline_values = calc_spline_values_for_nodes(y_table)

a, b = x_table[0][0], x_table[-1][0]
t_list = np.arange(a, b, 0.01)


int_x_list = tuple(spline(t, x_table, x_spline_values) for t in t_list)
int_y_list = tuple(spline(t, y_table, y_spline_values) for t in t_list)

ax.plot(int_x_list, int_y_list, label='spline')






dots_in, dots_out, x_min_max, y_min_max = fu(int_x_list, int_y_list, 1000)
x_min, x_max = x_min_max
y_min, y_max = y_min_max
s_rectangle = abs((x_max - x_min) * (y_max - y_min))
s_monte_carlo = Monte_Carlo_method(dots_in, dots_out, s_rectangle)

x_dot = tuple(x for x, y in dots_in)
y_dot = tuple(y for x, y in dots_in)

plt.scatter(x_dot, y_dot, color='coral', s=3)

x_dot = tuple(x for x, y in dots_out)
y_dot = tuple(y for x, y in dots_out)

plt.scatter(x_dot, y_dot, color='teal', s=3)

plt.show()

print(f'Симпсон {s_simpson  : .5f}, \n М-Карло {s_monte_carlo  : .5f}')
