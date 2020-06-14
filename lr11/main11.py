from numpy import arange
import math
from prettytable import PrettyTable

def tangent_and_chord(a, b, f, derivs) :
    first_deriv, second_deriv = derivs

    if first_deriv(a) * second_deriv(a) < 0 :
        # # метод хорд
        b = b - (b - a) * f(b) / (f(b) - f(a))
        # # метод касательных
        a = a - f(a) / first_deriv(a)
    else :
        # # метод хорд
        a = a - (b - a) * f(a) / (f(b) - f(a))
        # # метод касательных
        b = b - f(b) / first_deriv(b)
        
    return a, b

def combined_method(f, derivs, l, r, e, h) :
    intervals = tuple(arange(l, r + h / 2, h))
    intervals = tuple(zip(intervals[ : -1], intervals[1 : ]))

    answrs = []

    for a, b in intervals :
        # меняет ли функция знак на данном интервале?
        if f(a) * f(b) <= 0 :
            # значит на интервале есть решине
            # проверяем, если решение на конце отрезка
            if abs(f(a)) < e :
                # возможно, это решение уже было добавлено, как решение на конце предыдущего отрезка
                if len(answrs) > 0 and answrs[-1][2] != a :
                    answrs.append((a, b, a))
            elif abs(f(b)) < e :
                answrs.append((a, b, b))
            else :
                start, stop = a, b
                # если решение не в краевой точке
                while abs(a - b) > e :
                    a, b = tangent_and_chord(a, b, f, derivs)

                answrs.append((start, stop, (a + b) / 2))
    return answrs

def answr_to_str(answr, f) :
    table = PrettyTable(('Интервал', 'Решение', 'Невязка'))
    if len(answr) == 0 : 
        text = table.get_string() + '\nКорни не найдены на данном интервале с заданным шагом.\n'
    else :
        for a, b, n in answr :
            row = (f'[{a : .2f}, {b : .2f}]', f'{n : .5f}', f'{f(n) : .5f}')
            table.add_row(row)
        text = table.get_string() + '\n'  
    return text

f_list = [
    lambda x: x**3 - 2*x**2,
    lambda x: math.cos(x) ,
    lambda x: math.exp(-x) - math.sin(x)**2 / 2
]

f_alias = [
    'x**3 - 2*x**2',
    'cos(x) = 0',
    'exp^(-x) - sin(x)^2 / 2 = 0'

]
derivs_list = [
    (lambda x: 3*x**2 - 4*x, 
    lambda x: 6 * x - 4),
    (lambda x: -math.sin(x), 
    lambda x: -math.cos(x)),
    (lambda x: - math.exp(-x) - math.sin(2 * x) / 2, 
    lambda x: math.exp(-x) - math.cos(2 * x))
    ]

a_b_list = [
    (0, 4),
    (-9, 10),
    (0, 3)
]

e_l = (0.00001, )
h_list = [
    (3, 1),
    (5, 2, 1),
    (2, 0.5),
]
text = ''
for f, derivs, (a, b), steps, alias in zip(f_list, derivs_list, a_b_list, h_list, f_alias) :
    for h in steps :
        for e in e_l :
            text += f'Уравнение {alias}, на отрезке [{a : .2f}, {b : .2f}], с шагом {h} и точностью {e}.\n'
            answr = combined_method(f, derivs, a, b, e, h)
            text += answr_to_str(answr, f) + '\n'
    text += '\n'

with open('output11.txt', 'w', encoding='utf8') as file :
    file.write(text)