#!/usr/bin/env python

import numpy as np
from scipy.optimize import curve_fit

def diode_model(Is, n, Rs, current, Vt = 26e-3):
  return n * Vt * np.log( current / Is + 1) + current * Rs

def auto_fit_diode_model(current, voltage, Vt = 26e-3):
  least_err = np.inf
  found_solution = False
  for i in range(2, current.size):
    try:
      Is, n, Rs, err, r_pts, d_pts = fit_diode_model(current, voltage, i, Vt)
    except:
      continue
    found_solution = True
    #print Is, n, Rs
    if err < least_err:
      best_Is = Is
      best_n = n
      best_Rs = Rs
      best_r_pts = r_pts
      best_d_pts = d_pts
      least_err = err
  if found_solution:
    return best_Is, best_n, best_Rs, least_err, best_r_pts, best_d_pts
  else:
    raise RuntimeError('solution not found')

def fit_diode_model(current, voltage, initial_points = 1e6, 
                    Vt = 26e-3, n_guess = 1.5):
  N = min(current.size/2, initial_points)
  M = min(current.size/2, initial_points)

  # estimate Rs and (fixed) Vd from linear fit at high current points
  Pv = np.polyfit(current[-N:], voltage[-N:], 1)
  Rs = Pv[0]
  Vd = Pv[1]

  r_model_points = np.polyval(Pv, current)

  # estimate Is, n from non-linear fit at low current points
  b_scale_simple = n_guess * Vt
  c_scale_simple = np.log((np.exp(Vd / (n_guess * Vt)) - 1) / current[-N])
  def simple_diode(current, b, c):
    return b * b_scale_simple * np.log(np.exp(c * c_scale_simple) * current + 1)
   
  p_guess = [1, 1]
  try:
    p_opt, p_cov = curve_fit(simple_diode,
                             current[0:M], voltage[0:M],
                             p0 = p_guess)
  except:
    raise RuntimeError('intial diode fit failed')

  d_model_points = simple_diode(current, *p_opt)

  # estimate final values for all three parameters simultaneously
  def full_diode(current, b, c, a):
    return b * b_scale * np.log(c * c_scale * current + 1) + a*a_scale*current

  a_scale = Rs
  b_scale = p_opt[0] * b_scale_simple
  c_scale = np.exp(p_opt[1] * c_scale_simple)
  p_guess = [1, 1, 1]
  try:
    p_opt, p_cov = curve_fit(full_diode,
                             current, voltage,
                             p0 = p_guess)
  except:
    raise RuntimeError('final diode fit failed')

  
  err = np.sqrt(np.sum(np.square(voltage - full_diode(current, *p_opt))));

  Rs = p_opt[2] * a_scale
  n = p_opt[0] * b_scale / Vt
  Is = 1 / (p_opt[1] * c_scale)
  return Is, n, Rs, err, r_model_points, d_model_points
