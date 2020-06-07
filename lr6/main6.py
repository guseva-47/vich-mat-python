import math
import matplotlib.pyplot as plt

def polynom(k, x, table) :
    # (x - x_0) * ... * (x - x_(k-1)) * (x - x_(k+1)) * ... * (x - x_n)
    mul = 1
    for j, (x_j, _) in enumerate(table) :
        if j != k :
            mul *= x - x_j
    return mul

def calc_polynom_values_for_nodes(table) :
    return [polynom(k, x_k, table) for k, (x_k, _) in enumerate(table)]

# Возвращает значение интерпалиционного многочлена Лагранжа в точк еке x
def lagrange(x, table, polynom_values) :
    #           n    (      w(x)              )
    # L_n(x) = SUM  (  --------------- f(x_k)  )
    #          k=0   ( (x - x_k) w'(x)        )
    s = 0
    for k, d_om in enumerate(polynom_values) :
        f_x = table[k][1]
        s += f_x * (polynom(k, x, table) / d_om)
    return s

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

def drow_grafics(start_and_end, func, polynom_values, table, func_name) :
    plot_x_start, plot_x_end = start_and_end
    h = (plot_x_end - plot_x_start) / 1000
    x_list = (*(plot_x_start + i*h for i in range(1000)), plot_x_end)

    _, axes = plt.subplots()
    axes.set_title(func_name)
    axes.plot(x_list, tuple(map(func, x_list)), linewidth = 3)
    func = lambda x : lagrange(x, table, polynom_values)
    axes.plot(x_list, tuple(map(func, x_list)))

    plt.show()

def main() :
    # перечисление функций, которые будут обрабатываться
    exp_funk = lambda x : math.exp(-x**2)
    pow_funk = lambda x : math.pow(abs(x+1), math.cos(x))
    functions = [ abs, exp_funk, math.sin, pow_funk]
    func_alias = ['|x|', 'exp^(-x^2)', 'sin(x)', '|x+1|^cos(x)']

    text_result = ''
    
    byts = 0
    for func, alias in zip(functions, func_alias):
        text_result += 'Функция ' + alias + '\n'

        (nodes, x_values), byts = inp('input6.txt', byts)

        text_result += 'Узлы : ' + '  '.join(map(str, nodes)) + '\n'

        table = tuple((x, func(x)) for x in nodes)
        polynom_values = calc_polynom_values_for_nodes(table)

        for x in x_values:
            interpoled_y = lagrange(x, table, polynom_values)
            y = func(x)
            e = y - interpoled_y
            text_result += f'X : {x : 10.5f} \t'
            text_result += f'Y : {y : 10.5f} \t'
            text_result += f'L : {interpoled_y : 10.5f} \t'
            text_result += f'e : {e : 10.5f} \n'
        text_result += '\n'

    with open('output6.txt', 'w', encoding='utf8') as file : file.write(text_result)

    byts = 0
    for func, alias in zip(functions, func_alias):
        (nodes, _), byts = inp('input6.txt', byts)
        table = tuple((x, func(x)) for x in nodes)
        polynom_values = calc_polynom_values_for_nodes(table)

        start_and_end = (nodes[0], nodes[-1])
        drow_grafics(start_and_end, func, polynom_values, table, alias)



if __name__ == '__main__' :
    main()
