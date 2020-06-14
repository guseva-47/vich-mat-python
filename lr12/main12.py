# система уравнений
# система производных по каждой переменной каждого уравнения -- матрица якоби
# вектор приближений (просто вводится)

# while количетсво итераций в пределах разумного, какая-то норма болльше точности
#    вектор приближений прогоняем через все производные из матрицы якоби
#    получаем матрицу -- левая часть для решения гаусом
#    вектор приближений прогоняем через систему уровнений (входную). результаты со знаком минус -- правая часть для гауса
#  решить систему методом гауса, полученное реенеие -- дельта
# находим новое приближение: старое приближение плюс дельта x = [x[i] + delta[i] for i in range(len(x))]
# если норма дельты < e, выход

import numpy as np
from gauss import gauss
import math
from prettytable import PrettyTable

# totdo переименовать систему, якоби
def Newtons_method(equ_system, derivs, e, aproximation :np.ndarray) -> np.ndarray :
    iter_count = 0
    # фальшивое дельта, чтобы первый раз while сработал
    delta = np.fromiter((i + 1 for i in range(len(equ_system))), float, len(equ_system))
    
    while iter_count < MAX_ITER_COUNT and np.linalg.norm(delta) > e:
        # рассчет левой части системы уравнений
        A = tuple(tuple(d_xy(*aproximation) for d_xy in dF_list) for dF_list in derivs)
        A = np.array(A) # преобразование в numpy, т.к. функция м. гаусса работает с ним

        B = tuple(-equation(*aproximation) for equation in equ_system)
        B = np.array(B) # преобразование в numpy, т.к. функция м. гаусса работает с ним

        delta = gauss(A, B)
        aproximation += delta

        iter_count += 1
    
    assert iter_count > 0

    return aproximation, iter_count

def to_text(func_alias, iter_count, answr, neviazka, e, first_aprox) :
    text = 'Система уравнений. \n'
    for title in func_alias :
        text += title + ' = 0\n'
    text += '\n'

    variable_alias = ('x', 'y', 'z')

    text += 'Начальное приближение. \n'
    text += ', '.join((variable_alias[i] + f' = {first_aprox[i] : .5f}' for i in range(len(first_aprox)))) + '\n'

    text += f'e = {e}\n'

    table = PrettyTable(('Корни', 'Невязка'))
    
    roots = tuple(variable_alias[i] + f' = {answr[i] : .5f}' for i in range(len(answr)))
    neviazka = tuple(f'{nev : .5f}' for nev in neviazka)

    for root, nev in zip(roots, neviazka) :
        table.add_row((root, nev))
    text += table.get_string() + '\n'

    text += f'количество итераций = {iter_count}\n'
    
    text += '\n\n'
    return text

def main() :
    equations_system_list = (
        (lambda x, y: x**2 + (y - 2)**2 - 4, lambda x, y: (x-1)**2 / 4 + (y-2)**2 - 4),
        (lambda x, y: x**2 + (y - 2)**2 - 4, lambda x, y: (x-4)**2 + (y-1)**2 - 4)
    )
    derivs_list = (
        ((lambda x, y: 2*x, lambda x, y: 2*(y - 2)),
        (lambda x, y: (x - 1)/2, lambda x, y: 2*(y - 2))),
        ((lambda x, y: 2*x, lambda x, y: 2*(y - 2)),
        (lambda x, y: 2 * (x - 4), lambda x, y: 2*(y - 1)))
    )

    text = ''
    func_alias_list = (
        ('x**2 + (y - 2)**2 - 4', '(x-1)**2 / 4 + (y-2)**2 - 4'),
        ('x**2 + (y - 2)**2 - 4', '(x-4)**2 + (y-1)**2 - 4')
    )
    
    first_aproximations_list = (
        ((1, -1), ),
        #((0, 0), )
    )
    e_list = (0.00001, 0.1)
    for eq_system, derivs, alias, aproxs in zip(equations_system_list, derivs_list, func_alias_list, first_aproximations_list) :
        for first_aprox in aproxs :

            for e in e_list :
                np_first_aprox = np.array(first_aprox, dtype=float)
                answr, iter_count = Newtons_method(eq_system, derivs, e, np_first_aprox)
                if iter_count >= MAX_ITER_COUNT :
                    text += to_text(alias, iter_count, [], [], e, first_aprox) + 'Решение не найдено!\n\n\n'
                else :
                    neviazka = tuple(equation(*answr) for equation in eq_system)
                    text += to_text(alias, iter_count, answr, neviazka, e, first_aprox)
                
    with open('output12.txt', 'w', encoding='utf-8') as file :
        file.writelines(text)


if __name__ == "__main__":
    MAX_ITER_COUNT = 10000
    main()