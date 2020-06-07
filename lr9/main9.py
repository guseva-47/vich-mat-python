from numpy import arange
import matplotlib.pyplot as plt
import math
import datetime

# Метод прямоугольников
def middle_Riemann_sum(funk, a, b, h) :
    tmp = tuple(arange(a + h/2, b, h))
    tmp = map(funk, tmp)
    tmp = sum(tmp)
    tmp = h * tmp
    return tmp

# Метод трапеций
def trapezoidal_rule(funk, a, b, h) :
    tmp = tuple(arange(a, b + h/2, h))
    tmp = tuple(map(funk, tmp))
    tmp = sum(tmp[1 : -1]) + (tmp[0] + tmp[-1]) / 2
    tmp = h * tmp
    return tmp

# Метод Симпсона
def Simpson_method(funk, a, b, h) :
    even = tuple(arange(a + 2 * h, b, 2 * h))
    odd = tuple(arange(a + h, b, 2 * h))
    a_b = tuple(arange(a, b + h/2, h))
    a_b = (a_b[0], a_b[-1])

    even = map(funk, even)
    odd = map(funk, odd)
    a_b = map(funk, a_b)

    result = 2 * sum(even) + 4 * sum(odd) + sum(a_b)
    result = result * h / 3
    return result

# Автовыбор шага для любого метода
def auto_funk(funk, a, b, e, *_, middle = False, trapez = False, simpson = False) :
    method = None
    if middle : method = middle_Riemann_sum
    elif trapez : method = trapezoidal_rule
    elif simpson : method = Simpson_method
    else : return None, None

    last_result = None
    new_result = None
    h = 1
    # ограничение на число разбиений
    for count in range(10000) :
        new_result = method(funk, a, b, h)
        if last_result and abs(new_result - last_result) < e :
            break
        h /= 2
        last_result = new_result
    return new_result, count

# Отрисовка результатов для первого задания
def draw_task_1(result_list, error_list, reference, funk_alias) :
    def calc_labels_1(result_list, reference) :
        labels = ['Точный результат\n', 'М. средних\nпрямоугольников\n', 'М. трапеции\n', 'М. Симпсона\n']
        for i in range(len(labels)) :
            labels[i] += f'{result_list[i]: .5f}'
        return labels

    fig, (ax_res, ax_err) = plt.subplots(1, 2)
    x = tuple(i for i in range(len(result_list) + 1))
    result_list = [reference, *result_list]
    # левый график, график нйденных интегралов
    ax_res.barh(x, result_list, color='teal')
    ax_res.set_yticks(x)
    ax_res.set_yticklabels(calc_labels_1(result_list, reference))
    ax_res.set_title(funk_alias)
    # правый график, график погрешностей относительно правильного результата
    error_list = [0, *error_list]
    ax_err.barh(x, error_list, color='coral')
    ax_err.set_yticks(x)
    labels = tuple(map(lambda x: f'{x: .5f}', error_list))
    ax_err.set_yticklabels(labels)
    ax_err.set_title('Порешности результатов')

    fig.set_figwidth(15)
    now = datetime.datetime.now()
    fig.savefig(str(now.minute) + str(now.second) + '.png', bbox_inches='tight')

    plt.show()

# Отрисовка результатов задания 2
def draw_task_2(result_list, count_list, funk_alias) :
    def calc_labels_2(result_list, reference) :
        labels = ['М. средних\nпрямоугольников\n', 'М. трапеции\n', 'М. Симпсона\n']
        for i in range(len(labels)) :
            labels[i] += f'{result_list[i]: .5f}'
        return labels


    fig, (ax_res, ax_count) = plt.subplots(1, 2)
    x = tuple(i for i in range(len(result_list)))

    ax_res.barh(x, result_list, color='teal')
    ax_res.set_yticks(x)
    ax_res.set_yticklabels(calc_labels_2(result_list, result_list))
    ax_res.set_title(funk_alias)

    # правый график, график количества итераций для каждого метода
    ax_count.barh(x, count_list, color='orchid')
    ax_count.set_yticks(x)
    labels = tuple(map(str, count_list))
    ax_count.set_yticklabels(labels)
    ax_count.set_title('Количество итераций')

    fig.set_figwidth(15)
    now = datetime.datetime.now()
    fig.savefig(str(now.minute) + str(now.second) + '.png', bbox_inches='tight')

    plt.show()

# Путем проведения вычислительных экспериментов при постоянном шаге инте-
# грирования исследовать зависимость точности вычисления интеграла от выбо-
# ра метода и шага.
def first_task(references, funks, a, b, h, funks_alias) :
    if len(references) != len(funks) : return
    if len(references) != len(funks_alias) : return

    for i in range(len(references)) :
        ref, funk = references[i], funks[i]

        results = []
        results.append(middle_Riemann_sum(funk, a, b, h))
        results.append(trapezoidal_rule(funk, a, b, h))
        results.append(Simpson_method(funk, a, b, h))

        errors = tuple(map(lambda result: abs(ref - result), results))

        draw_task_1(results, errors, ref, funks_alias[i] + f', шаг = {h}')

# В случае автоматического выбора шага, исследовать зависимость количества
# шагов от метода и заданной точности.
def second_task(funks, a, b, e, funks_alias) :
    if len(funks) != len(funks_alias) : return

    for i in range(len(funks)) :
        funk = funks[i]
        results = []
        results.append(auto_funk(funk, a, b, e, middle=True))
        results.append(auto_funk(funk, a, b, e, trapez=True))
        results.append(auto_funk(funk, a, b, e, simpson=True))

        res = tuple(map(lambda x: x[0], results))
        counts = tuple(map(lambda x: x[1], results))

        draw_task_2(res, counts, funks_alias[i] + f', e = {e}')


def main() :
    # Функции от которых будет найден интеграл
    funk = [
        lambda x : x**2 - 2 * x**3 + 4 * x**4 - 2 * x**5,
        lambda x : -math.sin(2 * x),
        lambda x : 3 * x - 2 * x ** 2
    ]
    # Заведомо верные результаты интегралов
    references = [
        -1.066667,
        -0.82682,
        0.6666667,
    ]
    # текстовое отображение функций
    funk_alias = [
        'x^2 - 2 x^3 + 4 x^4 - 2 x^5',
        '-sin(2x) от 0 до 2',
        '3x - 2 x^2 от 0 до 2'
    ]

    for h in [0.01, 0.0001, 0.00001] :
        first_task(references, funk, 0, 2, h, funk_alias)
    
    for e in [0.01, 0.0001, 0.00001] :

        second_task(funk, 0, 2, e, funk_alias)


if __name__ == "__main__":
    main()
