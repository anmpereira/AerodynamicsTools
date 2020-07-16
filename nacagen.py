"""
Generate 4 digit NACA https://en.wikipedia.org/wiki/NACA_airfoil
Copyright (C) 2020  André Pereira

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
from scipy import interpolate


def _sampling(number_points: np.array, cosine_sampling: bool) -> np.array:
    if cosine_sampling:
        beta = np.linspace(0, np.pi, number_points)
        x = 0.5*(1 - np.cos(beta))
    else:
        x = np.linspace(0, 1, number_points)
    return x


class NACA4:
    """Note: all attributes are normalized to chord length
    """
    position_max_thickness = 0.30  # constant for NACA4 foils

    def __init__(self, code: int or str, chord_length: float, closed_te: bool, number_points=100, cosine_sampling=True):
        self.code = str(code)
        self.chord_length = chord_length
        self.closed_te = closed_te

        self.max_camber = float(self.code[0])/100
        self.position_max_camber = float(self.code[1])/10
        self.max_thickness = float(self.code[2:])/100
        self.le_radius = 1.1019*self.max_thickness**2

        if self.closed_te:
            self.a_i = (1.4845, -0.6300, -1.7580, 1.4215, -0.5075)
            self.te_gap = 0
        else:
            self.a_i = (1.4845, -0.6300, -1.7580, 1.4215, -0.518)
            self.te_gap = 2*np.arctan(1.16925*self.max_thickness)

        surface = self._generate_surface_points(number_points, cosine_sampling)
        x = surface['x']

        self._interpolators = {
            'yc': interpolate.interp1d(x, surface['yc'], kind='quadratic', assume_sorted=True),
            'dyc_dx': interpolate.interp1d(x, surface['dyc_dx'], kind='quadratic', assume_sorted=True),
            'xu': interpolate.interp1d(x, surface['xu'], kind='quadratic', assume_sorted=True),
            'yu': interpolate.interp1d(x, surface['yu'], kind='quadratic', assume_sorted=True),
            'xl': interpolate.interp1d(x, surface['xl'], kind='quadratic', assume_sorted=True),
            'yl': interpolate.interp1d(x, surface['yl'], kind='quadratic', assume_sorted=True)
        }

    def get_foil_data(self, nr_samples: np.array, cosine_sampling=True) -> dict:
        chord_coordinates = _sampling(nr_samples, cosine_sampling)
        c = self.chord_length
        interp = self._interpolators

        return {
            'x': chord_coordinates*c,
            'yc': interp['yc'](chord_coordinates)*c,
            'dyc_dx': interp['dyc_dx'](chord_coordinates)*c,
            'xu': interp['xu'](chord_coordinates)*c,
            'yu': interp['yu'](chord_coordinates)*c,
            'xl': interp['xl'](chord_coordinates)*c,
            'yl': interp['yl'](chord_coordinates)*c,
        }

    def _generate_surface_points(self, number_points: int or float, cosine_sampling: bool) -> np.array:
        # setup data
        a0, a1, a2, a3, a4 = self.a_i
        m = self.max_camber
        p = self.position_max_camber
        number_points = int(number_points)
        x = _sampling(number_points, cosine_sampling)

        # calculate thickness function
        yt = self.max_thickness*(a0*x**0.5 + a1*x + a2*x**2 + a3*x**3 + a4*x**4)

        # calculate camber and its derivative
        if m == 0:  # uncambered foil
            yc = np.zeros(len(x))
            dyc_dx = np.zeros(len(x))

        else:  # cambered foil
            yc = np.where(x < p,
                          (m/(p**2))*(2*p*x - x**2),
                          (m/((1 - p)**2))*(1 - 2*p + 2*p*x - x**2))

            dyc_dx = np.where(x < p,
                              (2*m/(p**2))*(p - x),
                              (2*m/((1 - p)**2))*(p - x))

        # local camber angle
        theta = np.arctan(dyc_dx)

        # upper and lower coordinates
        xu = x - yt*np.sin(theta)
        yu = yc + yt*np.cos(theta)
        xl = x + yt*np.sin(theta)
        yl = yc - yt*np.cos(theta)

        return {
            'x': x,
            'yc': yc,
            'dyc_dx': dyc_dx,
            'xu': xu,
            'yu': yu,
            'xl': xl,
            'yl': yl
        }


class Display(object):
    def __init__(self):
        self.plt = plt
        self.h = []
        self.label = []
        self.fig, self.ax = self.plt.subplots()
        self.plt.axis('equal')
        self.plt.xlabel('x')
        self.plt.ylabel('y')
        self.ax.grid(True)

    def plot(self, X, Y, label=''):
        h, = self.plt.plot(X, Y, '-', linewidth=1)
        self.h.append(h)
        self.label.append(label)

    def show(self):
        self.plt.axis((-0.1, 1.1) + self.plt.axis()[2:])
        self.ax.legend(self.h, self.label)
        self.plt.show()


if __name__ == "__main__":
    # main()
    import matplotlib.pyplot as plt
    foil = NACA4('2212', 10, False)
    data = foil.get_foil_data(100)
    print(data['x'], data['yc'])

    fig, ax = plt.subplots()
    plt.plot(data['xu'], data['yu'], 'b')
    plt.plot(data['xl'], data['yl'], 'b')
    plt.plot(data['x'], data['yc'], 'r')
    ax.grid()
    ax.set_aspect('equal')
