

def progonka(A, B, C, D):
    '''
        Решает СЛАУ с трехдиагональной матрицей методом прогонки.
        - vec_abcd - список, в котором каждый элемент - кортеж из соответствующих 
        элементов векторов A, B, C, D.
            - A - диагональ под главной, первый элемент равен нулю;
            - B - главная диагональ;
            - C - диагональ над главной, последний элемент равен нулю;
            - D - столбец правой части системы.
        - возвращает вектор X - решение системы.
    '''
    
    vec_alf, vec_bet = __forward_move(zip(A, B, C, D))
    return __backward_move(vec_alf, vec_bet)

def __forward_move(vec_abcd):
    ''' Прямой проход метода прогонки '''

    vec_abcd = map(lambda v: (v[0], -v[1], v[2], v[3]), vec_abcd)

    a, b, c, d = next(vec_abcd)
    vec_alf, vec_bet = [c/b], [-d/b]

    for (a, b, c, d), alf, bet in zip(vec_abcd, vec_alf, vec_bet):
        vec_alf.append(c/(b-a*alf))
        vec_bet.append((a*bet-d)/(b-a*alf))

    return vec_alf, vec_bet

def __backward_move(vec_alf, vec_bet):
    ''' Обратный проход метода прогонки '''

    factors = zip(reversed(vec_alf), reversed(vec_bet))
    alf, bet = next(factors)

    rev_x = [bet]

    for (alf, bet), x in zip(factors, rev_x):
        rev_x.append(alf*x + bet)

    rev_x.reverse()
    return rev_x
