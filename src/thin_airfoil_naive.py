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


# TODO: split vectorized functions from single value functions to reduce confusion
# TODO: calculate p_s_angle only once for the whole mesh? Yes, like vortex panel method
def p_s_angle(x_norm, y_norm, theta):
    """Returns the angle on the plane between point (x_norm, y_norm) and point (theta) on the chord

    Args:
        x_norm:
        y_norm:
        theta:

    Returns:
        the angle in radians, counter-clockwise from the positive x semi-axis
    """
    return np.arctan2(y_norm, x_norm - 0.5*(1 - np.cos(theta)))


def get_grad(potential):
    """Returns the gradient with the discontinuity fixed

    Args:
        potential:

    Returns:

    """
    grad = np.gradient(potential)
    gx = grad[1]
    gy = grad[0]

    # fix discontinuity
    shape = gx.shape  # format (nr_y, nr_x)

    # fix line in the middle
    line = shape[0]//2
    side_grad = np.gradient(potential[line + 1:line + 3, :])[0][-1, :]
    gy[line] = side_grad

    if shape[0] % 2 == 0:  # if even nr_y, fix the other middle line
        line = shape[0]//2 - 1
        side_grad = np.gradient(potential[line - 3:line - 1, :])[0][-1, :]
        gy[line] = side_grad

    return gx, gy


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

    if isinstance(alpha, float) or isinstance(alpha, int):
        ai[0] = alpha - (1/np.pi)*integrate.quad(integrand_generator(0), 0, np.pi)[0]
        return ai

    else:
        aii = np.zeros((len(ai), len(alpha)))
        a0 = -(1/np.pi)*integrate.quad(integrand_generator(0), 0, np.pi)[0]
        for i, a in enumerate(alpha):
            ai[0] = a + a0
            aii[:, i] = ai
        return aii


def gamma_sin_norm(norm_foil: nacagen.NACA4, max_order: int, alpha: float):
    """Returns a function for the normalized vortex sheet strength multiplied by sin(theta), where theta is the habitual
    transform from x = c/2(1 - cos theta). The function returns a value for a given theta
    The constant of normalization is V_inf

    Args:
        norm_foil:
        max_order:
        alpha:

    Returns:

    """
    ai = _get_a_coeff(norm_foil, max_order, alpha)
    n = np.arange(1, len(ai))

    def f(theta):
        if isinstance(theta, float):
            factor1 = ai[0]*(1 + np.cos(theta))
            factor2 = np.sum(ai[1:]*np.sin(n*theta))*np.sin(theta)
            return 2*(factor1 + factor2)
        else:
            gamma_i = np.zeros(len(theta))
            factor1 = ai[0]*(1 + np.cos(theta))
            for i, t in enumerate(theta):
                factor2 = np.sum(ai[1:]*np.sin(n*t))*np.sin(t)
                gamma_i[i] = 2*(factor1[i] + factor2)
            return gamma_i

    return f


def velocity_pot(norm_foil: nacagen.NACA4, nr_x, nr_y, max_order: int, alpha: float):
    # create the gamma_sin function
    alpha = np.deg2rad(alpha)
    gamma_sin_f = gamma_sin_norm(norm_foil, max_order, alpha)

    # create the solve grid
    x_norm = np.linspace(-1, 2, nr_x)
    y_norm = np.linspace(-1, 1, nr_y)
    xv, yv = np.meshgrid(x_norm, y_norm)
    potential = np.zeros((nr_y, nr_x))

    for i in range(nr_x):
        for j in range(nr_y):
            x = xv[j, i]
            y = yv[j, i]

            vortex = -1/(4*np.pi)*integrate.quad(lambda theta: gamma_sin_f(theta)*p_s_angle(x, y, theta),
                                                 0, np.pi)[0]
            free = x*np.cos(alpha) + y*np.sin(alpha)

            potential[j, i] = vortex + free

    return xv, yv, potential


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
    # '2408'
    foil = nacagen.NACA4('2412', number_points=400)
    # alphas = np.linspace(-5, 10, 10)
    # data = get_data_alpha(foil, alphas)
    # chord = 10

    # foil_data = foil.get_foil_data(200, chord)
    # fig, ax = plt.subplots()
    # plt.plot(foil_data['xu'], foil_data['yu'], 'b')
    # plt.plot(foil_data['xl'], foil_data['yl'], 'b')
    # plt.plot(foil_data['x'], foil_data['yc'], 'r')
    # ax.grid()
    # ax.set_aspect('equal')
    #
    # plt.figure()
    # plt.plot(alphas, data['cl'])
    # plt.plot(alphas, data['cm_c_4'])
    # plt.grid()

    # print(_get_a_coeff(foil, 10, 10))
    # plt.figure()
    # plt.plot(gamma_sin_norm(foil, 10, 10)(np.linspace(0, np.pi, 100)))

    plt.figure()
    potential = velocity_pot(foil, 200, 200, 20, -2)
    pot = potential[2]
    grad = get_grad(pot)
    grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)*100  # why 100?
    plt.contourf(potential[0][0, :], potential[1][:, 0], grad_mag, levels=50)
    plt.colorbar()
    foil_data = foil.get_foil_data(100, 1)
    str_points = np.array([np.zeros(120), np.linspace(-1, 1, 120)]).transpose()
    plt.streamplot(potential[0][0, :], potential[1][:, 0], grad[0], grad[1], density=40, arrowstyle='-',
                   start_points=str_points, color='k')

    plt.plot(foil_data['x'], foil_data['yc'], 'r')
