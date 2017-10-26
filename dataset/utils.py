""" Contains helper functions """
import copy

def copy1(data):
    if isinstance(data, tuple):
        out = tuple(copy1_list(data))
    elif isinstance(data, list):
        out = copy1_list(data)
    elif isinstance(data, dict):
        out = copy1_dict(data)
    else:
        raise TypeError("Unsupprted type '{}'".format(type(data)))
    return out

def copy1_list(data):
    return [copy.copy(item) for item in data]

def copy1_dict(data):
    return dict((key, copy.copy(item)) for key, item in data.items())
