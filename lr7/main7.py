import math
import matplotlib.pyplot as plt
import numpy as np
import datetime

def polynom(k, x, table) :
    # (x - x_0) * ... * (x - x_(k-1)) * (x - x_(k+1)) * ... * (x - x_n)
    mul = 1
    for j, (x_j, _) in enumerate(table) :
        if j != k :
            mul *= x - x_j
    return mul

def calc_polynom_values_for_nodes(table) :
    return [polynom(k, x_k, table) for k, (x_k, _) in enumerate(table)]

# Возвращает значение интерпалиционного многочлена Лагранжа в точке x
def lagrange(x, table, polynom_values) :
    #           n    (      w(x)              )
    # L_n(x) = SUM  (  --------------- f(x_k)  )
    #          k=0   ( (x - x_k) w'(x)        )
    s = 0
    for k, d_om in enumerate(polynom_values) :
        f_x = table[k][1]
        s += f_x * (polynom(k, x, table) / d_om)
    return s

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

def inp(filename,  byts = 0) :
    with open(filename, 'r') as file :
        file.seek(byts)
        tmp = []
        while True :
            line = file.readline()
            if line == '' or line == '\n' : break

            tmp.append(tuple(map(float, line.split())))
        
        byts = file.tell()
    return tuple(tmp), byts

def drow_grafics(start_and_end, func_list, func_name) :
    x_list = np.arange(*start_and_end, 0.01)

    fig, axes = plt.subplots()
    axes.set_title(func_name)
    for func, width in zip(func_list, range(len(func_list), 0, -1)):
        axes.plot(x_list, tuple(map(func, x_list)), linewidth = width)

    now = datetime.datetime.now()
    file_name = str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)
    fig.savefig(file_name + '.png', bbox_inches='tight')
    plt.show()

def main() :
    # перечисление функций, которые будут обрабатываться
    exp_funk = lambda x : math.exp(-x**2)
    pow_funk = lambda x : math.pow(abs(x+1), math.cos(x))
    x_pow9_func = lambda x: x**9

    functions = [ abs, exp_funk, math.sin, pow_funk, x_pow9_func]
    func_alias = ['|x|', 'exp^(-x^2)', 'sin(x)', '|x+1|^cos(x)', 'x^9']

    # сравнение многочлена Лагранжа и сплайна
    byts = 0
    for func, alias in zip(functions, func_alias):
        (nodes, *_), byts = inp('input7.txt', byts)
        table = tuple((x, func(x)) for x in nodes)

        polynom_values = calc_polynom_values_for_nodes(table)
        lagrange_func = lambda x: lagrange(x, table, polynom_values)

        spline_values = calc_spline_values_for_nodes(table)
        spline_func = lambda x: spline(x, table, spline_values)

        start_and_end = (nodes[0], nodes[-1])
        func_list = (func, lagrange_func, spline_func)
        drow_grafics(start_and_end, func_list, alias)
    

    def get_spline_func(table, spline_values):
        return lambda x: spline(x, table, spline_values)

    # влияние погрешности
    func, alias = functions[1], func_alias[1]
    (nodes, *_), byts = inp('input7.txt', byts)

    nodes_cpy = nodes

    errors = [0, 0.45, 0.9]
    for ind in range(2, 5): 
        func_list = [func]
        for err in errors:
            nodes = (*nodes_cpy[:ind], nodes_cpy[ind] + err, *nodes_cpy[ind+1:])
            table = tuple((x, func(x)) for x in nodes)

            spline_values = calc_spline_values_for_nodes(table)
            spline_func = get_spline_func(table, spline_values)

            func_list.append(spline_func)

        start_and_end = (nodes[0], nodes[-1])
        drow_grafics(start_and_end, func_list, alias)
    

    # различные способы параметризации таблицы

    x_func = lambda t: math.cos(t) + math.cos(2*t)
    y_func = lambda t: math.sin(t) - math.sin(2*t)
    
    t_list = np.linspace(0, 2*math.pi, 8)

    table = tuple((x_func(t), y_func(t)) for t in t_list)

    def case1(table):
        x_table = tuple((i, x) for i, (x, _) in enumerate(table))
        y_table = tuple((i, y) for i, (_, y) in enumerate(table))
        return x_table, y_table
    
    def case2(table):
        d_table = tuple((next_x-x, next_y-y) for (next_x, next_y), (x, y) in zip(table[1:], table[:-1]))

        t_list = [0]
        for d_x, d_y in d_table:
            d_t = math.sqrt(d_x**2 + d_y**2)
            t_list.append(t_list[-1] + d_t)
        
        x_table = tuple((t, x) for t, (x, _) in zip(t_list, table))
        y_table = tuple((t, y) for t, (_, y) in zip(t_list, table))
        
        return x_table, y_table
    
    def case3(table):
        x_table = tuple((math.sqrt(i), x) for i, (x, _) in enumerate(table))
        y_table = tuple((math.sqrt(i), y) for i, (_, y) in enumerate(table))

        x_table = tuple((i**(1/7), x) for i, (x, _) in enumerate(table))
        y_table = tuple((i**(1/7), y) for i, (_, y) in enumerate(table))
        return x_table, y_table
    

    t_list = np.arange(t_list[0], t_list[-1], 0.01)
    x_list = tuple(map(x_func, t_list))
    y_list = tuple(map(y_func, t_list))

    cases = (case1, case2, case3)
    aliases = ('case1', 'case2', 'case3')

    fig, axes = plt.subplots(1, len(cases))
    for case, axis, alias in zip(cases, axes, aliases):
        x_table, y_table = case(table)

        x_spline_values = calc_spline_values_for_nodes(x_table)
        y_spline_values = calc_spline_values_for_nodes(y_table)

        t_list = np.arange(x_table[0][0], x_table[-1][0], 0.01)

        axis.set_title(alias)

        axis.plot(x_list, y_list, label='origin')

        int_x_list = tuple(spline(t, x_table, x_spline_values) for t in t_list)
        int_y_list = tuple(spline(t, y_table, y_spline_values) for t in t_list)
        axis.plot(int_x_list, int_y_list, label='spline')

    now = datetime.datetime.now()
    file_name = str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)
    fig.savefig(file_name + '.png', bbox_inches='tight')
    plt.show()
    
if __name__ == '__main__' :
    main()