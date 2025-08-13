import numpy as np
from typing import Tuple, Dict, List
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


"""
NOt used anywhere.
"""

class MiscVisUtils():
    """Miscalenious vis utils that don't fit anywhere but are to nice to throw out.
    
    
    """


    # following code is taken from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    @classmethod
    def confidence_ellipse(cls, mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

        Returns
        -------
        matplotlib.patches.Ellipse
        """

        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                        facecolor=facecolor, **kwargs)

        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mean[0]

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mean[1]

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)


    @classmethod
    def draw_em_results(cls, points, means, covariances):
        fig, ax = plt.subplots(figsize=(12, 12))

        # plot 1
        # ACHTUNG: Always only plot plot 1 or plot 2. Comment the other out!!!!!!!
        ax = ax
        x = points[:, 0]
        y = points[:, 1]
        colors = ["salmon", "lightgreen", "powderblue", "slateblue", "pink"]
        accent_colors = ["red", "lime", "cyan", "blue", "hotpink"]
        ax.scatter(x, y, c=colors[1], s=0.05)
        K = means.shape[0]
        for i in range(K):
            cls.confidence_ellipse(means[i], covariances[i], ax=ax, n_std=3.0, edgecolor='r', linestyle='--')
    
        plt.show()