#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tools for Optics11 Chiaro indenter

@author: Nicolas Bruot (https://www.bruot.org/hp/)
@date: 10 September 2018
"""

import numpy as _numpy
import datetime as _datetime
import matplotlib.pyplot as _plt


def _parse_tab_sep_line(line, required_elems):
    """Parses a tab-separated line and checks the values of required fields

    Args:
        line: line as str
        required_elems: Set of the shape ((col_idx1, val1),
                                          (col_idx2, val2), ...).

    Returns:
        elems: All the (tab-separated) columns of the line.

    Raises:
        ValueError: When any of the column values does not match the expected
            value in `required_elems`.
    """

    line = line.rstrip('\r\n')
    elems = line.split('\t')
    for col, val in required_elems:
        if elems[col] != val:
            raise ValueError('Cannot parse file: expected "%s" text.' % val)

    return elems


def _parse_single_field(line, req_name):
    """Parses a line of the type "field_name\tvalue\""""

    elems = _parse_tab_sep_line(line, ((0, req_name), ))

    return elems[1]


def _check_single_line(line, expected_str):
    if line.rstrip('\r\n') != expected_str:
        raise ValueError('Cannot parse file: expected "%s" text.' % expected_str)


class Indentation(object):
    """Indentation experiment data reader"""

    def __repr__(self):
        return '<Indentation: %s>' % self.name


    def __str__(self):
        return self.name


    def __init__(self, path):
        """Instantiates an Indentation object

        Args:
            path: Path of the .txt file containing the indentation data from
                the experiment with the Chiaro indenter.
        """

        self.path = path

        with open(path, 'r') as f:
            # Line 1
            elems = _parse_tab_sep_line(f.readline(),
                                        ((0, 'Date'), (2, 'Time'), (4, 'Status')))
            date_str = '%s %s' % (elems[1], elems[3])
            date_fmt = '%d/%m/%Y %H:%M:%S'
            self.date = _datetime.datetime.strptime(date_str, date_fmt)
            self.status = elems[5]
            # Line 2
            self.name = f.readline().rstrip('\r\n')
            # Line 3
            elems = _parse_tab_sep_line(f.readline(),
                                        (
                                            (0, 'Scan (#)'),
                                            (2, 'X (#)'),
                                            (4, 'Y (#)'),
                                            (6, 'Indentation (#)'),
                                         ))
            self.n_scans = int(elems[1])
            self.n_x = int(elems[3])
            self.n_y = int(elems[5])
            self.n_indentations = int(elems[7])
            # Line 4
            self.x_pos = float(_parse_single_field(f.readline(),
                                                   'X-position (um)'))
            # Line 5
            self.y_pos = float(_parse_single_field(f.readline(),
                                                   'Y-position (um)'))
            # Line 6
            self.z_pos = float(_parse_single_field(f.readline(),
                                                   'Z-position (um)'))
            # Line 7
            self.z_surf = float(_parse_single_field(f.readline(),
                                                    'Z surface (um)'))
            # Line 8
            self.piezo_pos = float(_parse_single_field(f.readline(),
                                                       'Piezo position (nm) (Measured)'))
            f.readline()

            # Line 10
            self.k = float(_parse_single_field(f.readline(),
                                               'k (N/m)'))
            f.readline()

            # Line 12
            self.tip_radius = float(_parse_single_field(f.readline(),
                                                        'Tip radius (um)'))
            # Line 13
            self.calib_factor = float(_parse_single_field(f.readline(),
                                                          'Calibration factor'))
            # Line 14
            self.wavelength = float(_parse_single_field(f.readline(),
                                                        'Wavelength (nm)')) * 1e-3
            # Line 15
            val = _parse_single_field(f.readline(), 'Auto find surface')
            if val == 'Not used in single indent mode':
                self.auto_find_surf = None
            else:
                raise ValueError('Auto find surface: Value not supported yet.')
            # Line 16
            val = _parse_single_field(f.readline(), 'dX before scan (um)')
            if val == 'Only available in auto find surface mode':
                self.dx_before_z_scan = None
            else:
                raise ValueError('dX before scan (um): Value not supported yet.')
            f.readline()

            # Line 18
            self.piezo_pos_setpoint_at_start = float(_parse_single_field(f.readline(),
                                                                         'Piezo position setpoint at start (nm)'))
            f.readline()

            # Line 20
            _check_single_line(f.readline(),
                               'Piezo Indentation Sweep Settings (Relative to piezo position setpoint at start):')
            # Lines 21 through 20 + n_pts
            lines = []
            while True:
                line = f.readline()
                if not line.rstrip('\r\n'):
                    break
                lines.append(line)
            n_pts = len(lines)
            self.dt_prof = _numpy.empty(n_pts)
            self.z_prof = _numpy.empty(n_pts)
            for i in range(n_pts):
                elems = _parse_tab_sep_line(lines[i],
                                            ((0, 'D[Z%d] (nm)' % (i + 1)),
                                             (2, 't[%d] (s)' % (i + 1))))
                self.z_prof[i] = float(elems[1]) * 1e-3
                self.dt_prof[i] = float(elems[3])

            # Line 22 + n_pts
            self.P_max = float(_parse_single_field(f.readline(),
                                                   'P[max] (uN)')) * 1e-6
            # Line 23 + n_pts
            self.D_max = float(_parse_single_field(f.readline(),
                                                   'D[max] (nm)')) * 1e-3
            # Line 24 + n_pts
            self.D_final = float(_parse_single_field(f.readline(),
                                                     'D[final] (nm)')) * 1e-3
            # Line 25 + n_pts
            self.D_max_min_final = float(_parse_single_field(f.readline(),
                                                             'D[max-final] (nm)')) * 1e3
            # Line 26 + n_pts
            self.slope = float(_parse_single_field(f.readline(),
                                                   'Slope (N/m)'))
            # Line 27 + n_pts
            self.E_eff = float(_parse_single_field(f.readline(),
                                                   'E[eff] (Pa)'))
            # Line 28 + n_pts
            self.E_half_v = float(_parse_single_field(f.readline(),
                                                      'E[v=0.50] (Pa)'))
            f.readline()

            # Line 35: columns headers
            _check_single_line(f.readline(),
                               'Time (s)\tLoad (uN)\tIndentation (nm)\tCantilever (nm)\tPiezo (nm)\tAuxiliary')

            # The rest of the file corresponds to indentation data
            data = _numpy.genfromtxt(f)
            self.t, self.load, self.z_i, self.z_c, self.z_p, self.aux = data.T
            self.z_i *= 1e-3
            self.z_c *= 1e-3
            self.z_p *= 1e-3


    def plot_profile(self):
        """Plots the profile of the displacement-controlled indentation"""

        n = len(self.dt_prof)
        t = _numpy.zeros(n + 1)
        for i in range(n):
            t[i + 1] = t[i] + self.dt_prof[i]
        z = _numpy.zeros(n + 1)
        z[1:] = self.z_prof
        _plt.plot(t, z)
        _plt.xlabel('t (s)')
        _plt.ylabel('z (µm)')
        _plt.gca().invert_yaxis()


    def plot_displacements(self):
        """Plots the displacements of the piezo, cantilever and computed indentation against time"""

        _plt.plot(self.t, self.z_p,
                  self.t, self.z_c,
                  self.t, self.z_i)
        _plt.legend(('Piezo', 'Cantilever', 'Indentation'))
        _plt.xlabel('t (s)')
        _plt.ylabel('z (µm)')
        _plt.gca().invert_yaxis()


    def plot_load(self):
        """Plots the computed load against time"""

        _plt.plot(self.t, self.load)
        _plt.xlabel('t (s)')
        _plt.ylabel('Load (µN)')


    def plot_load_vs_z(self):
        """Plots the computed load against the indentation"""

        _plt.plot(self.z_i, self.load)
        _plt.xlabel('z (µm)')
        _plt.ylabel('Load (µN)')
