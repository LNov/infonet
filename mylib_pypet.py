import numpy as np


def print_leaves(traj, node_name):
    try:
        node = traj.f_get(node_name)
    except AttributeError as e:
        print(e)
    else:
        if node.v_is_group:
            if node.f_has_leaves():
                for leave in node.f_iter_leaves():
                    if leave.f_has_range():
                        print(leave.v_full_name, np.unique(leave.f_get_range()), sep=' : ')
                    else:
                        print(leave.v_full_name, leave.f_val_to_str(), sep=' : ')
            else:
                print('node \'{}\' has no leaves'.format(node_name))
        else:
            if node.f_has_range():
                print(node.v_full_name, np.unique(node.f_get_range()), sep=' : ')
            else:
                print(node.v_full_name, node.f_val_to_str(), sep=' : ')

