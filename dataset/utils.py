""" Utilities functions """


def get_del(adict, key, default):
    """ Return key from a dict if exists, otherwise default """
    if key in adict.keys():
        val = adict[key]
        del adict[key]
    else:
        val = default
    return val
