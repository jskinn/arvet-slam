import typing
import matplotlib.pyplot as plt
import seaborn as sns


NAME = 'plot_abs_error_distribution'


def get_required_columns() -> typing.Set[str]:
    return {'abs_error_length'}


def plot(dataframe, output=''):
    data = dataframe['abs_error_length']

    fig, axes = plt.subplots(1, 1, figsize=(9, 9))
    sns.distplot(data, hist=False, color="r", kde_kws={"shade": True}, ax=axes)

    # if output:
    #     fig.save()

    return fig
