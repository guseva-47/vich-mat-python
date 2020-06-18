import numpy as np


def eigenvalues(A, e):
    MAX_STEPS = 100

    size = A.shape[0]
    R = (A**2).sum(axis=1) - A.diagonal()**2

    # сумма квадратов недиагональных элементов
    nondiag_sum = R.sum()

    B = np.array(A)
    D = np.array([[0]*size for _ in range(size)], float)
    for i in range(size):
        D[i, i] = 1

    for i in range(MAX_STEPS):
        # вычисление индексов k, l
        k, _ = max(enumerate(R), key=lambda r: abs(r[1]))
        l, _ = max(enumerate(B[k]), key=lambda b: abs(b[1]) if b[0] != k else -1)

        # проверка погрешности
        if nondiag_sum < e:
            eig_list = np.array([B[i, i] for i in range(size)])
            return eig_list, D, i

        # вычисление alpha и beta
        if B[k, k] == B[l, l]:
            alpha = np.sqrt(0.5)
            beta = np.sqrt(0.5)
        else:
            m = 2*B[k, l]/(B[k, k] - B[l, l])
            alpha = np.sqrt((1 + 1/np.sqrt(1+m**2)) / 2)
            beta = np.sign(m) * np.sqrt((1 - 1/np.sqrt(1+m**2)) / 2)

        # D = D * U
        tmp_k = D[:, k]*alpha + D[:, l]*beta
        tmp_l = -D[:, k]*beta + D[:, l]*alpha
        D[:, k], D[:, l] = tmp_k, tmp_l

        # B = U^(-1) * B * U
        tmp_k = B[:, k]*alpha + B[:, l]*beta
        tmp_l = -B[:, k]*beta + B[:, l]*alpha
        B[:, k], B[:, l] = tmp_k, tmp_l

        tmp_k = B[k]*alpha + B[l]*beta
        tmp_l = -B[k]*beta + B[l]*alpha
        B[k], B[l] = tmp_k, tmp_l

        # обновление R и суммы квадратов недиагональных элементов
        tmp_k = B[k]**2
        tmp_k = tmp_k[:k].sum() + tmp_k[k+1:].sum()

        tmp_l = B[l]**2
        tmp_l = tmp_l[:l].sum() + tmp_l[l+1:].sum()

        nondiag_sum += tmp_k + tmp_l - R[k] - R[l]
        R[k], R[l] = tmp_k, tmp_l

    eig_list = np.array([B[i, i] for i in range(size)])
    return eig_list, D, MAX_STEPS


def get_nevyazka(a_mx, eigenvalues, eigenvectors):
    # TODO: переделать проверку невязки
    return abs(a_mx @ eigenvectors - eigenvalues * eigenvectors).max(axis=0)


if __name__ == "__main__":
    # TODO: взять другие примеры
    matricies = [
        np.array([
            [1, 2],
            [2, 5],
        ], float),

        np.array([
            [1, 2, 3],
            [2, 5, 7],
            [3, 7, 9],
        ], float),

        np.array([
            [5,  7,  6,  5],
            [7, 10,  8,  7],
            [6,  8, 10,  9],
            [5,  7,  9, 10],
        ], float),
    ]

    for a_mx in matricies:
        amounts = []
        residuals = []
        for i in range(10):
            err = 10**-i
            values, vectors, amount = eigenvalues(a_mx, err)
            r = get_nevyazka(a_mx, values, vectors)

            print(f'{err} {amount}, {r}', values, vectors, '', sep='\n\n')
            amounts.append(amount)
            residuals.append(abs(r).max())

        print(amounts)
        print('[', ', '.join(map(lambda r: f'{r:0.2e}', residuals)), ']', sep='')
