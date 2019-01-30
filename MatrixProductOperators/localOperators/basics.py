import numpy as np

def annih_up(site, length):

    I = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]

    a = [[0, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 0]]

    I = np.array([[I]])

    a = np.array([[a]])

    output = site * [I] + [a] + (length - site - 1) * [I]

    # output[0] = output[0][0]
    # output[-1] = output[-1][0]

    # print(output[0].shape)
    # print(output[-1].shape)
    # print(output[int(site / 2)].shape)

    return output # no transposing since it is used with my index convention....

#
# def unity(length):
#
#     I = [[1, 0, 0, 0],
#          [0, 1, 0, 0],
#          [0, 0, 1, 0],
#          [0, 0, 0, 1]]
#
#     I = np.asarray([[I]])
#
#     a = np.asarray([[a]])
#
#     return [I] + (length - 2) * [I]