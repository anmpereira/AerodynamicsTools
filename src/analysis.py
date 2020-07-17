"""
Analyse airfoil data
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
from src import nacagen
import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt


def _get_a_coeff(norm_foil: nacagen.NACA4, max_order: int, alpha):
    data_ = norm_foil.get_foil_data(200, 1)
    dzc_dx_f = interpolate.interp1d(data_['x'], data_['dyc_dx'])

    def integrand_generator(n):
        """Generates an integrand used to determine the coefficient An

        Args:
            n: order of the a parameter

        Returns:
            a function that generates the value of the integrand for an input theta
        """
        def f(theta):
            x = 0.5*(1 - np.cos(theta))
            return dzc_dx_f(x)*np.cos(n*theta)
        return f

    # independent of alpha
    ai = [0]*(max_order + 1)
    for i in range(1, max_order + 1):
        ai[i] = (2/np.pi)*integrate.quad(integrand_generator(i), 0, np.pi)[0]

    if isinstance(alpha, float):
        ai[0] = alpha - (1/np.pi)*integrate.quad(integrand_generator(0), 0, np.pi)[0]
        return ai

    else:
        aii = np.zeros((len(ai), len(alpha)))
        a0 = -(1/np.pi)*integrate.quad(integrand_generator(0), 0, np.pi)[0]
        for i, a in enumerate(alpha):
            ai[0] = a + a0
            aii[:, i] = ai
        return aii


def get_data_alpha(norm_foil: nacagen.NACA4, alpha) -> dict:
    alpha = np.deg2rad(alpha)
    a = _get_a_coeff(norm_foil, max_order=2, alpha=alpha)
    cl = 2*np.pi*(a[0] + 0.5*a[1])
    cl_slope = 2*np.pi
    alpha_l0 = alpha - cl/(2*np.pi)
    cm_c_4 = np.pi/4*(a[2] - a[1])
    cm_le = -(cl/4 - cm_c_4)
    x_cp = np.where(cl == 0, np.inf, 1/4*(1 + np.pi*(a[1] - a[2])/cl))

    return {
        'cl': cl,
        'cl_slope': cl_slope,
        'alpha_l0': alpha_l0,
        'cm_c_4': cm_c_4,
        'cm_le': cm_le,
        'x_cp': x_cp
    }


if __name__ == '__main__':
    foil = nacagen.NACA4('2408', number_points=300)
    alphas = np.linspace(-5, 10, 10)
    data = get_data_alpha(foil, alphas)
    plt.plot(alphas, data['cl'])
    plt.plot(alphas, data['cm_c_4'])
    plt.grid()

