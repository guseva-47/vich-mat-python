import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime

def calc_k1(system, x, y_list, h) :
    return [h * f(x, *y_list) for f in system]

def calc_k2(system, x, y_list, h, k1) :
    tmp_y = tuple(y_list[i] + k1[i] * 0.5 for i in range(len(k1)))
    x = x + h * 0.5
    return calc_k1(system, x, tmp_y, h)

def calc_k3(system, x, y_list, h, k2) :
    return calc_k2(system, x, y_list, h, k2)

def calc_k4(system, x, y_list, h, k3) :
    tmp_y = tuple(y_list[i] + k3[i] for i in range(len(k3)))
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

def nested_auto_step(dif_equ_system, y0_list, a_b, e, MIN_H=0.0001) :
    def calc_local_err(k1, k2, k3, k4):
        calc_Egorov = lambda k1, k2, k3, k4 : abs(2 * (k1 - k2 - k3 + k4) / 3.0)
        return max([calc_Egorov(k1[i], k2[i], k3[i], k4[i]) for i in range(len(k1))])

    a, b = a_b
    new_a = a
    h = (b - a) / 2
    
    x_list = [a]
    all_y = [y0_list]

    MIN_E = e / 2**5
    
    while new_a < b :
        # если из-за шага интервал выйдет за пределы, то нужно его уменьшить до максимально допустимого
        if new_a + h > b : h = abs(b - a)

        while True :
            # y_list -- y_n (последнее верно вычисленное y)
            y_list = all_y[-1]
            a_b = (new_a, new_a + h)
            _, all_y_n, all_kn = Runge_Kutta_methods(dif_equ_system, y_list, a_b, h)

            err = calc_local_err(*all_kn)

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

def to_drow(x_list, y_all, a_b, exact_f_s, f_title) :
    fig, ax = plt.subplots()
    ax.set_title(f_title)

    for i in range(len(y_all[0])) :
        ax.plot(x_list, tuple(y[i] for y in y_all), linewidth = 3)
    
    ex_x_list = np.arange(*a_b, 0.01)
    for f in exact_f_s :
        ax.plot(ex_x_list, tuple(map(f, ex_x_list)), label='reference')
    now = datetime.now()
    fig.savefig('plot/'+str(now.minute).zfill(2) + str(now.microsecond) + '.png', bbox_inches='tight')


def global_err_norm(x_list, all_y, ref_system) :
    all_ref_y = []
    for x in x_list :
        tmp = [f(x) for f in ref_system]
        all_ref_y.append(tmp)
    
    maxs = []
    for y_l, ref_y_l in zip(all_y, all_ref_y) :
        maxs.append(max(abs(y_l[i] - ref_y_l[i]) for i in range(len(y_l))))
    return max(maxs)

def main() :
    f_system_list = (
        [lambda x, y: 3 * x**2 + (y - x**3)],
        [lambda x, y: 3 * x**2 - 10 * (y - x**3)] #,
        # [lambda x, y1, y2: -2*y1 + 4*y2, lambda x, y1, y2: -y1 + 3*y2]

    )
    syst_title_list = (
        'y\' = 3 x + y - x**3',
        'y\' = 3 x - 10 (y - x**3)',
        'y1\'= -2*y1 + 4*y2 \ny2\'= -y1 + 3y2'
    )
    reference_f_s_list = (
        [lambda x: x**3],
        [lambda x: x**3],
        [lambda x: 4 * math.exp(-x) - math.exp(2 * x),
            lambda x: math.exp(-x) - math.exp(2 * x)]
    )
    a_b_list = (
        (0, 0.5),
    )

    h_list = (0.1, 0.01, 0.001)
    e_list = (0.1, 0.01, 0.001)
    MIN_H = 0.000001
    text = ''

    for system, ref_syst, syst_title in zip(f_system_list, reference_f_s_list, syst_title_list) :
        for a_b in a_b_list:
            a, b = a_b
            y0_list = tuple(f(a) for f in ref_syst)

            text += syst_title + f'\tна интервале [{a}, {b}]\n' + 'Постоянный шаг: \n'
            for h in h_list :
                x_list, all_y, _ = Runge_Kutta_methods(system, y0_list, a_b, h)
                
                title = syst_title + '\n' + f'Шаг = {h}'
                to_drow(x_list, all_y, a_b, ref_syst, title)

                glob_e = global_err_norm(x_list, all_y, ref_syst)
                text += f'\tШаг = {h}'.ljust(15) + f'Норма глоб.погр-ти = {glob_e : 0.10f}' + '\n'
            text += '\n'

            text += syst_title + f'\tна интервале [{a}, {b}]\n' + 'Автоматический выбор шага: \n'
            for e in e_list:
                x_list, all_y = nested_auto_step(system, y0_list, a_b, e, MIN_H)

                title = syst_title + '\n' + f'e = {e}'
                to_drow(x_list, all_y, a_b, ref_syst, title)

                glob_e = global_err_norm(x_list, all_y, ref_syst)
                text += f'\te = {e}'.ljust(15) + f'Норма глоб.погр-ти = {glob_e : 0.8f}' + '\n'

            text += '\n\n'

    with open('output13.txt', 'a', encoding='utf8') as file :
        file.write(text)

    print(text)

if __name__ == "__main__":
    main()
    # syst_title = 'u\' + 30u = 0'
    # system = [lambda x, y: -30 * y]
    # a, b = 0, 1
    # y0_list = [1]
    # h_list = [1/10, 1/11]

    # text = syst_title + f'\tна интервале [{a}, {b}]\n' + 'Постоянный шаг: \n'
    # for h in h_list :
    #     x_list, all_y, _ = Runge_Kutta_methods(system, y0_list, (a,b), h)
        
    #     title = syst_title + '\n' + f'Шаг = {h}'
    #     to_drow(x_list, all_y, (a,b), [], title)

    # text += '\n'
    
    
    # with open('output13.txt', 'w', encoding='utf8') as file :
    #     file.write(text)

    # print(text)