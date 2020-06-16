from runge_kutta_4 import nested_auto_step
from math import cos
from numpy import arange
import matplotlib.pyplot as plt
from datetime import datetime

# метод написан для решения системы из двух уравнений, причем первое уравнение (v) имеет начальное условие (решение в точке a, v(a) или v0),
# а второе (u) не имеет, но имеет решение на правой границе интервала (u(b))
# l_r -- интервал, внутри которого (как ожидается) лежит u(a)
# a_b -- интервал 
def shooting_method(dif_equ_system, a_b, l_r, va, ub, h, e) :

    def f_interval() :
        l, r = l_r
        l_r_list = arange(l, r, h)

        # решение ОДУ методом Рунге-Кутта с автошагом, для первого приближения (v(a), u(a))
        # возвращает только решение u(b) 
        def take_result(ul) :
            _, all_sollution = nested_auto_step(dif_equ_system, (va, ul), a_b, e)
            return all_sollution[-1][1]

        u_left = take_result(l)
        genrator = (take_result(r_i) for r_i in l_r_list[1 : ])

        for i, u_right in enumerate(genrator, 1) :
            # если решение лежит в left или right, то == 0, если между ними, то <0
            if (u_left - ub) * (u_right - ub) <= 0 :
                # промежуток найден
                left, right = l_r_list[i - 1], l_r_list[i]
                return left, right
            
            u_left = u_right
        
        return None, None

    # левая и правая граница укороченного интервала, на котором находится решение = u(a)
    new_l, new_r = f_interval()
    if new_l is None or new_r is None : return None, None

    # уточнение результата бинарным поиском. С ограничением на количество итераций
    MAX_COUNT_ITER = 10000
    for i in range(MAX_COUNT_ITER) :
        middle  = (new_l + new_r) / 2
        x_list, all_sollution = nested_auto_step(dif_equ_system, (va, middle), a_b, e)
        _, y_name = all_sollution[-1]
        if abs(y_name - ub) < e :
            break
        elif y_name > ub :
            new_r = middle
        else:
            new_l = middle

    if i == MAX_COUNT_ITER : x_list, all_sollution = None, None
    
    return x_list, all_sollution

def to_drow(x_list, y_all) :
    fig, ax = plt.subplots()

    for i in range(len(y_all[0])) :
        ax.plot(x_list, tuple(y[i] for y in y_all))

    now = datetime.now()
    fig.savefig('plot/'+str(now.minute).zfill(2) + str(now.microsecond) + '.png', bbox_inches='tight')
    plt.show()

def main() :
    dif_equ_system = (
        lambda x, u, v : v,
        lambda x, u, v : -0.3 * cos(v)
    )
    a_b = (0, 1)
    l_r = (0, 1)
    va = 0
    ub = 0
    h = 0.1
    e = 0.000001

    x_list, all_sollution = shooting_method(dif_equ_system, a_b, l_r, va, ub, h, e)
    to_drow(x_list, all_sollution)

if __name__ == "__main__":
    main()