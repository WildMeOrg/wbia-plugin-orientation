# -*- coding: utf-8 -*-
def main():  # nocover
    import wbia_orientation

    print('Looks like the imports worked')
    print('wbia_orientation = {!r}'.format(wbia_orientation))
    print('wbia_orientation.__file__ = {!r}'.format(wbia_orientation.__file__))
    print('wbia_orientation.__version__ = {!r}'.format(wbia_orientation.__version__))


if __name__ == '__main__':
    """
    CommandLine:
       python -m wbia_orientation
    """
    main()
