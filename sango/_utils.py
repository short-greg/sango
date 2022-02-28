
def coalesce(x, y):
    return y if x is None else x


def vals(cls):
    """Use to retrieve all class members

    Yields:
        _type_: _description_
    """

    try:
        annotations = cls.__annotations__
    except AttributeError:
        annotations = {}
    d  = getattr(cls, '__dict__', {})

    for var in [x for x in d.keys() if not x.startswith('__')]:
        annotation = annotations.get(var, None)
        val = getattr(cls, var)
        yield var, annotation, val

