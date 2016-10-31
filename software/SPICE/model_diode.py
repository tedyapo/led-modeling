#!/usr/bin/env python

import numpy as np
import diode_model as dm

for color in ['amber', 'blue', 'cool_white', 'cyan', 'deep_red',
              'full_spectrum', 'green', 'natural_white',
              'orange', 'pink', 'red', 'royal_blue', 'uv', 'warm_white',
              'white', 'yellow']:
  data = np.loadtxt('chanzon_3W_LEDs/chanzon_3W_%s.dat' % color)
  current = data[:, 1]
  voltage = data[:, 2]
  intensity = data[:, 3]

  Is, n, Rs, Err, r_pts, d_pts = dm.auto_fit_diode_model(current, voltage)

  # estimate polynomial model for relative intensity
  Pi = np.polyfit(current, intensity, 5)
  Pi = Pi / np.polyval(Pi, 100e-3)
  
  print '.MODEL chanzon_3W_%s D(Is=%e, n=%e, Rs=%e)' % (color, Is, n, Rs)

  model_voltage = dm.diode_model(Is, n, Rs, current)
  np.savetxt('chanzon_3W_%s_plot.dat' % color,
             np.array([current, voltage,
                       model_voltage, r_pts, d_pts]).transpose(),
             header = '# i v model_v r_model_v d_model_v')

