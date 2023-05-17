import matplotlib.pyplot as plt
# mport seaborn as sns
import random
import numpy as np
from typing import Union


class Grapher:
    def __init__(self):
        """
        - Stores the graphs and shows them on demand
        """
        self._plt = plt

    # Helper Functions #
    def get_rnd_color(self):
        """
        - Generates random RGB values
        :return: RGB values
        """
        r = random.random()
        g = random.uniform(0, 0.4)
        b = random.random()

        return r, g, b

    # / Helper Functions #

    def make_curve(self, title: str, label: Union[str, None], x_label: str, y_label, color: str,
                   marker: Union[str, None], y, additional_ys=(), additional_label=(), single_point_data=None,
                   vline=0):
        """ Single points: ((x_0,y_0, 'label'_0)...(x_n, y_n, 'label'_n)) """
        if len(additional_ys) != len(additional_label):
            print('Error. Number of add. labels does not match number of add. data sets')
            return 1

        if vline != 0:
            self._plt.axvline(x=vline, color='darkred', lw=0.3, label='Phase Change')

        self._plt.title(title)
        self._plt.xlabel(x_label)
        self._plt.ylabel(y_label)
        self._plt.plot(range(len(y)), y, linewidth=0.5, linestyle='solid', marker=marker,
                       color=color, markerfacecolor=color, label=label)

        r_0, g_0, b_0 = self.get_rnd_color()

        for y_add, label in zip(additional_ys, additional_label):
            self._plt.plot(range(len(y_add)), y_add, linewidth=0.5, linestyle='solid', marker='o',
                           color=(r_0, g_0, b_0), markerfacecolor=(r_0, g_0, b_0), label=label)

        if single_point_data:
            for single_point in single_point_data:
                r, g, b = self.get_rnd_color()
                x, y, single_label = single_point
                self._plt.plot([x], [y], linewidth=0.5, linestyle='solid', marker='x',
                               color=(r, g, b), markerfacecolor=(r, g, b), label=single_label)
        self._plt.legend()
        self._plt.figure()
        return 0

    def make_curve_subplot(self, title: str, label: Union[str, None], x_label: str, y_label, color: str,
                   marker: Union[str, None], y, additional_ys=(), additional_labels=(), single_point_data=None,
                   vline=0, sd=None):
        """ Single points: ((x_0,y_0, 'label'_0)...(x_n, y_n, 'label'_n)) """
        if len(additional_ys) != len(additional_labels):
            print('Error. Number of add. labels does not match number of add. data sets')
            return 1

        fig, ax1 = self._plt.subplots()

        if vline != 0:
            ax1.axvline(x=vline, color='darkred', lw=0.3, label='Phase Change')

        self._plt.title(title)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.plot(range(len(y)), y, linewidth=0.5, linestyle='solid', marker=marker,
                 color=color, markerfacecolor=color, label=label)
        ax1.tick_params(axis='y', labelcolor=color)
        self._plt.legend()

        # Instantiate second axes that share the x-axis
        ax2 = ax1.twinx()
        # ax2.set_xlabel(x_label)
        # ax2.set_ylabel(y_label)
        for y_add, add_label in zip(additional_ys, additional_labels):
            ax2.plot(range(len(y_add)), y_add, linewidth=0.5, linestyle='solid', marker='o',
                           color='red', markerfacecolor='red', label=add_label)
        ax2.tick_params(axis='y', labelcolor='red')

        if single_point_data:
            for single_point in single_point_data:
                r, g, b = self.get_rnd_color()
                x, y, single_label = single_point
                self._plt.plot([x], [y], linewidth=0.5, linestyle='solid', marker='x',
                               color=(r, g, b), markerfacecolor=(r, g, b), label=single_label)

        if sd:
            additional_ys, = additional_ys
            sd_y, sd_y_add = sd
            ax1.fill_between(range(len(y)), y-sd_y, y+sd_y, color='gray', alpha=0.2)
            ax2.fill_between(range(len(additional_ys)), additional_ys - sd_y_add, additional_ys + sd_y_add,
                             color='pink', alpha=0.2)
            # ax2.fill_between(range(len(additional_ys)), additional_ys-sd_y_add, additional_ys+sd_y_add,
            #                  color='pink', alpha=0.2, where=additional_ys-sd_y_add >50)

        fig.tight_layout()
        self._plt.legend(bbox_to_anchor=(0.842, 0.87), loc='upper right')
        self._plt.figure()
        return 0

    def make_bar(self, label_x: str, title: str, heights, pos='center', color='blue', iter_start=0,
                 xlim=Union[None, tuple], ylim=Union[None, tuple]):
        x = []
        for h in range(len(heights)):
            label_i = f'{label_x}{h+iter_start}'
            x.append(label_i)
        self._plt.bar(x=x, height=heights, width=0.3, align=pos, color=color)

        # Optional Lim
        if xlim:
            x_a, x_b = xlim
            self._plt.xlim(x_a, x_b)
        if ylim:
            y_a, y_b = ylim
            self._plt.ylim(y_a, y_b)
        self._plt.title(title)
        self._plt.figure()

    def make_freq_couter(self, data, title:str, n_bins: int, width, bin_range: list,
                         color='lightskyblue', pos='edge', x_label='', y_label='', save_file=None) -> int:
        """
        - Plots a histogram where the count is shown for each "bin"
        :param data: Flattened iterable data, i.e. np.ndarray
        :param title: Titel of plot
        :param n_bins: Number of bins
        :param width: Width of bins/bars
        :param bin_range: [Lowest bin, highest bin]
        :param color: Bar color
        :param pos: Alignemnt
        :param x_label: Label for x-axis
        :param y_label: label for y-axis
        :param save_file: Toogles whether plot is shown in run-mode or saved to file
        :return: Error indication
        """
        if len(bin_range) != 2:
            raise ValueError('bin_range must be given as a list: [a,b]')
        if n_bins <= 1:
            raise ValueError('n_bins must be > 1!')
        max_length = bin_range[1] - bin_range[0]
        interval_length = max_length / n_bins
        intervals = np.linspace(bin_range[0], bin_range[1], n_bins+1)
        bins = []
        for bin in range(len(intervals)-1):
            bins.append([])

        for data_i in data:
            for idx, bin in enumerate(bins):
                if data_i > intervals[idx] and data_i <= intervals[idx+1]:
                    # if data_i <= intervals[idx+1]:
                    bins[idx].append(data_i)

        interval_heights = []
        len_bins = len(bins)
        len_bins_half = int(len_bins / 2)
        for bin in bins:
            height_i = len(bin)
            interval_heights.append(height_i)
        labels = [bin_range[0]] + ['']*(len_bins-2) + [bin_range[1]]
        labels[len_bins_half] = str(intervals[len_bins_half])[:3]
        intervals_ = intervals[:-1]
        self._plt.bar(x=intervals_, height=interval_heights, width=width, align=pos, color=color)
        self._plt.xlim(bin_range[0], bin_range[1])
        # self._plt.xticks(intervals_)
        self._plt.xticks(intervals_, labels=labels)
        # # Set tick to 10 nd round to 2 decimals
        # self._plt.xticks(np.around(np.linspace(bin_range[0], bin_range[1], 10)), 2)
        self._plt.ylabel(y_label)
        self._plt.xlabel(x_label)
        self._plt.title(title)
        self._plt.grid(visible=True, which='major', axis='y')
        if save_file:
            self._plt.savefig(save_file)
        self._plt.figure()

        return 0

    def make_2d_scatter(self, axis0, axis1, labels, cmap='Spectral', s=5, title='',
                        norm=None, alpha=None, data=None, save_file=None):
        scatter2d = self._plt.scatter(axis0, axis1, c=labels, cmap=cmap, s=s, norm=norm,
                                      alpha=alpha, data=data)
        handles, labels = scatter2d.legend_elements()
        self._plt.title(title)
        self._plt.legend(handles, labels, loc='upper right', title='Labels')
        if save_file:
            self._plt.savefig(save_file)
        self._plt.figure()

    def make_3d_scatter(self, axis0, axis1, axis2, labels, cmap='Spectral', s=5, title='', norm=None,
                        alpha=None, data=None):

        self._plt.title(title)
        fig, ax = self._plt.subplots()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(axis0, axis1, axis2, c=labels, cmap=cmap, s=s, norm=norm, alpha=alpha,
                   data=data)
        fig.tight_layout()
        # self._plt.figure()

    def show_plots(self):
        self._plt.show()
        return 0

    def close_plots(self):
        self._plt.close('all')
