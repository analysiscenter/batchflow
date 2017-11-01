""" Contains classes to handle batch data components """


class ComponentDescriptor:
    """ Class for handling one component item """
    def __init__(self, component, default=None):
        self._component = component
        self._default = default

    def __get__(self, instance, cls):
        if instance.data is None:
            return self._default
        elif instance.pos is None:
            return instance.data[self._component]
        pos = instance.pos[self._component]
        data = instance.data[self._component]
        return data[pos] if data is not None else self._default

    def __set__(self, instance, value):
        if instance.pos is None:
            new_data = list(instance.data) if instance.data is not None else [None for _ in instance.components]
            new_data[self._component] = value
            instance.data = tuple(new_data)
        else:
            pos = instance.pos[self._component]
            instance.data[self._component][pos] = value


class BaseComponentsTuple:
    """ Base class for a component tuple """
    components = None

    def __init__(self, data=None, pos=None):
        if isinstance(data, BaseComponentsTuple):
            self.data = data.data
        else:
            self.data = data
        if pos is not None and not isinstance(pos, list):
            pos = [pos for _ in self.components]
        self.pos = pos

    def __str__(self):
        s = ''
        for comp in self.components:
            d = getattr(self, comp)
            s += comp + '\n' + str(d) + '\n'
        return s

    def as_tuple(self, components=None):
        """ Return components data as a tuple """
        components = tuple(components or self.components)
        return tuple(getattr(self, comp) for comp in components)



class MetaComponentsTuple(type):
    """ Class factory for a component tuple """
    def __init__(cls, *args, **kwargs):
        _ = kwargs
        super().__init__(*args, (BaseComponentsTuple,), {})

    def __new__(mcs, name, components):
        comp_class = super().__new__(mcs, name, (BaseComponentsTuple,), {})
        comp_class.components = components
        for i, comp in enumerate(components):
            setattr(comp_class, comp, ComponentDescriptor(i))
        globals()[comp_class.__name__] = comp_class
        return comp_class
