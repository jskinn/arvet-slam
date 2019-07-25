import typing


NAME = 'plot_absolute_trajectory'


def get_required_columns() -> typing.Set[str]:
    return {
        'x',
        'y',
        'z'
    }

def plot(dataframe, display=False, output=''):
