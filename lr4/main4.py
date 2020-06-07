import numpy as np
from math import sqrt
from gauss import gauss

# Составить программу, которая позволяет: 
# а) найти методом итераций два наибольших по модулю собственных числа матрицы и соответствующих им собственные вектора; 
# б) методом обратных итераций найти собственное число, ближайшее к заданному числу 𝜆0. 
# Исходные данные — матрица, точность, число 𝜆0 и начальный вектор должны читаться из файла,
# а результаты расчетов помещаться в файл. Предусмотрите вывод числа итераций, которые пришлось совершить для нахождения заданной точности.
ITER_MAX_COUNT = 1000

def inp_data(filename) :
    with open(filename, 'r') as file :
        # чтение матрицы, символы отделены табуляциями, после матрицы пустая строка
        matrix = []
        for line in file :
            if line == '' or line == '\n' : break

            tmp = [float(n) for n in line.split('\t')]
            matrix.append(tmp)
        
        # чтение точности, одна строка, отделенная пустой строкой от матрицы
        line = file.readline()
        e = float(line)

        # следующая строка -- число 𝜆0 (без пустой строки)
        line = file.readline()
        l0 = float(line)

        # следующая строка -- начальный вектор, символы отделены табуляцией
        line = file.readline()
        v0 = [float(n) for n in line.split('\t')]

        return matrix, e, l0, v0
    return None, None, None, None


def norm_vector(x) :
    return sqrt(np.dot(x, x))

def residual(A, l, x):
    return A@x - l*x

# универсальный поиск лямбда, в зависимости от параметров
def find_l_multi(x, A, e, *, l0 = None, g1 = None, e1 = None) :
    new_x = None
    old_x = np.copy(x)
    go = True
    time_to_time_period = 1

    #
    if e1 is not None and g1 is not None :
        old_x -= (np.dot(old_x, g1) / np.dot(e1, g1)) * e1

    # первый шаг
    if l0 == None :
        new_x = A @ old_x
    else :
        A = A - l0 * np.identity(len(A))
        new_x = gauss(A, old_x)
        
    new_l = np.dot(new_x, old_x) / np.dot(old_x, old_x)

    count_of_iter = 0

    while go :
        count_of_iter += 1
        old_x = new_x
        old_l = new_l
        # нормируем вектор время от времени
        if count_of_iter % time_to_time_period == 0 :
            norm_x = norm_vector(old_x)
            old_x = np.divide(old_x, norm_x)
            #
            if e1 is not None and g1 is not None :
                old_x -= (np.dot(old_x, g1) / np.dot(e1, g1)) * e1

        if l0 == None :
            new_x = A @ old_x
        else :
            new_x = gauss(A, old_x)
        new_l = np.dot(new_x, old_x) / np.dot(old_x, old_x)
        have_answr = abs(old_l - new_l) < e

        go = count_of_iter < ITER_MAX_COUNT and not have_answr
    return new_x, count_of_iter, new_l

def str_help_funk(l, x,*_, r = None, count_iter = None) :
    text = f"Количество итераций = {count_iter}\n" if count_iter != None else ""
    text += f"Невязка = {r}\n" if r is not None else ""
    text += f"Собственное число = {l}\n"
    text += f"Собственный вектор = {x}\n"
    return text

# наибольшее по модулю лямюда
def highest_abs_l(x0, A, e) :
    text = "Нахождение первого наибольшего по модулю лямбда. \n"
    x, iter_count, l = find_l_multi(x0, A, e)
    x /= norm_vector(x)

    need_repeat = False
    r = None
    if iter_count >= ITER_MAX_COUNT :
        need_repeat = True
        text += "Ответ не был найден, количество итераций слишком высоко.\n\n"
    else :
        # 
        norm_x = norm_vector(x)
        e1 = np.divide(x, norm_x)
        # проверям невязку
        r = residual(A, l, e1)
        if norm_vector(r) > e:
            need_repeat = True
            text += f"Норма невязки высока = {norm_vector(r)} > {e}\n\n"
    
    if need_repeat :
        C = 6
        text += f"Нахождение для B = A + {C}E\n"
        B_plus = A + C * np.identity(len(A))
        x_b_plus, count_iter_b_p, l_b_p  = find_l_multi(x, B_plus, e)
        l_b_p -= C
        x_b_plus /= norm_vector(x_b_plus)
        r_b_p = residual(A, l_b_p, x_b_plus)

        text += str_help_funk(l_b_p, x_b_plus, count_iter=count_iter_b_p, r=r_b_p)
        text += '\n'

        text += f"Нахождение для B = A - {C}E\n"
        B_minus  = A - C * np.identity(len(A))
        x_b_minus, count_iter_b_m, l_b_m = find_l_multi(x, B_minus, e)
        l_b_m += C
        x_b_minus /= norm_vector(x_b_minus)
        r_b_m = residual(A, l_b_m, x_b_minus)

        text += str_help_funk(l_b_m, x_b_minus, count_iter=count_iter_b_m, r=r_b_m)
        text += "\n"
    else :
        text += str_help_funk(l, x, count_iter=iter_count, r=r)

    return text

# второе наибольшее по модулю лямбда
def second_highest_l(x0, A, e) :
    text = "Нахождение второго наибольшего по модулю лямбда. \n"
    e1, *_ = find_l_multi(x0, A, e)
    A_transpon = np.transpose(A)
    g1, *_ = find_l_multi(x0, A_transpon, e)
    x, count_iter, l = find_l_multi(x0, A, e, g1=g1, e1=e1)

    r = residual(A, l, x)    
    text += str_help_funk(l, x, count_iter=count_iter, r=r)
    text += "\n"
    return text
    
# ближайшее к l0
def nearest_l (x0, A, e, l0) :
    text = f"Нахождение лямбда ближайшего к {l0}. \n"
    x, count_iter, near_l = find_l_multi(x0, A, e, l0=l0)
    near_l = l0 + 1 / near_l
    
    r = residual(A, near_l, x)
    text += str_help_funk(near_l, x, count_iter=count_iter, r=r)
    text += '\n'
    return text

def main() :
    matrix, e, l0, v0 = inp_data("input4.txt")
    A = np.array(matrix)
    start_vector = np.array(v0)
    text = "Матрица: \n"
    text += str(A) + '\n'
    text += "Начальный вектор х : " + str(start_vector) + '\n'
    text += "l_0 : " + str(l0) + "\n"
    text += "Точность : " + str(e) + "\n\n"
    
    print(text)
    with open("output4.txt", "w", encoding="UTF-8") as file :
        file.write(text)

    menu = "Частичная проблема собственных чисел.\n"
    menu += "1. Первое наибольшее по модулю собственное число.\n" 
    menu += "2. Второе наибольшее по модулю собственное число.\n"
    menu += f"3. Собственное чсило близкое к {l0}. \n"
    menu += "0. Выход.\n"
    while True :
        print(menu)
        result = ""
        key = input().split('\n')[0]
        key = int(key) if key.isdigit() else -1
        if key == 0 :
            break
        elif key == 1 :
            result = highest_abs_l(start_vector, A, e)
        elif key == 2 :
            result = second_highest_l(start_vector, A, e)
        elif key == 3 :
            result = nearest_l(start_vector, A, e, l0)
        
        print(result)
        with open("output4.txt", "a", encoding="UTF-8") as file :
            file.write(result)

if __name__ == "__main__":
    main()