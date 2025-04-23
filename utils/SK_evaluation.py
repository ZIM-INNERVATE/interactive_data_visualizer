import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

# class to set configuration parameters for the manuver
class config:
    def __init__(self):
        self.alarm_high = 48
        self.warning_high = 44
        self.alarm_low = 32
        self.warning_low = 36
        self.freq = 100
        self.target = 40
        self.goal_t = 180

    def update_config(self, new_values):
        for key, value in new_values.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")

# functions to compute evaluation for radius
class modul_R:

    def __init__(self, radius, config):
        self.global_param = config #config()
        self.radius = radius
        self.R_low_limit = self.global_param['target'] - (self.global_param['warning_high'] - self.global_param['target']) / 2
        self.R_upper_limit = self.global_param['target'] + (self.global_param['warning_high'] - self.global_param['target']) / 2
        self.avg_radius = 0
        self.avg_radius_conv = 0
        self.area_outside = 0
        self.max_area = 0
        self.area_percent = 0

    # Axuluary function to compute modul_R:
    def compute_meas_range_R(self):
        # radius = [startTime, endTime-5s]
        # meas_radius = self.radius[:-5 * self.global_param['freq']]
        meas_radius = self.radius

        # Convolution-Based Gleitender Mittelwert
        kernel = np.ones(100) / 100
        self.avg_radius_conv = np.convolve(meas_radius, kernel, mode='same')
        self.avg_radius_conv[0:50] = 40*np.ones((50))
        self.avg_radius_conv[len(self.avg_radius_conv)-50:len(self.avg_radius_conv)] = 40*np.ones((50))
              
        # Gleitender Mittelwert
        avg_radius = meas_radius.rolling(window=100).mean()
        self.avg_radius = [np.nan if np.isnan(point) else point for point in avg_radius]
        
        # distances to points above and below upper and low ranges   
        values_outside_limits = [x - self.R_upper_limit if (x > self.R_upper_limit) 
                                 else self.R_low_limit - x if (x < self.R_low_limit) 
                                 else 0 for x in avg_radius]
        # area above and below
        self.area_outside = np.trapz(values_outside_limits)
        # precentage of this area with respect to the whole area
        self.max_area = (self.R_upper_limit - self.R_low_limit) * len(values_outside_limits) * 0.01
        print("Max area: ", round(self.max_area, 2), "m2")
        print("Area outside: ", round(self.area_outside, 2), "m2")
        print("Area percent: ", round(self.area_percent, 2), "%")
        return self.area_percent

    def compute_mean_R(self):
        meas_radius = self.radius[:-5 * self.global_param['freq']]
        mean_R = np.mean(meas_radius)
        if mean_R > self.global_param['alarm_high'] or mean_R < self.global_param['alarm_low']:
            R_mean_percent = 0
        elif (self.global_param['target'] - 1) < mean_R < (self.global_param['target'] + 1):
            R_mean_percent = 100
        else:
            R_mean_percent = (1 - (mean_R - self.global_param['target'] + 1) / \
                          (self.global_param['alarm_high'] - self.global_param['target'] + 1)) * 100 if mean_R >= 40 else \
                          (1 - (self.global_param['target'] - 1 - mean_R) / (
                          self.global_param['target'] - 1 - self.global_param['alarm_low'])) * 100
        return R_mean_percent

    def compute_module_R(self):
        self.modul_R = 0.6 * self.compute_meas_range_R() + 0.4 * self.compute_mean_R()
        return self.modul_R

# functions to compute evaluation for accelaration
class modul_dH:

    def __init__(self, lenkradwin, config):
        self.global_param = config #config()
        self.lenkradwin = lenkradwin
        self.modul_dH = 0
        self.avg_lenkradwin = []
        self.meas_time = []
        self.fit_curve = []

    def fit_curve_dH(self):
        # lenkradwin = [startTime, endTime-10s]
        meas_lenkradwin = self.lenkradwin[:-10 * self.global_param['freq']]
        meas_time = np.arange(len(meas_lenkradwin))
        # Gleitender Mittelwert
        avg_lenkradwin = meas_lenkradwin.rolling(window=20).mean()
        avg_lenkradwin = avg_lenkradwin.dropna()
        self.avg_lenkradwin = avg_lenkradwin
        meas_time = meas_time[len(meas_time)-len(avg_lenkradwin):]
        poly_values = np.polynomial.polynomial.Polynomial.fit(meas_time, avg_lenkradwin, 3)
        self.fit_curve = poly_values(meas_time)
        self.meas_time = meas_time
         # model evaluation
        rmse = mean_squared_error(avg_lenkradwin, self.fit_curve)
        curve_dH = np.interp(rmse, [1, 5], [100, 0])
        return curve_dH

    def compute_mean_dH(self):
        # compute lenkscurve using First two seconds for lenkradwin
        signalArray = self.lenkradwin[0:2 * self.global_param['freq']]
        # lenkscurve = sum(signalArray) / abs(sum(signalArray))
        lenkscurve = np.sign(sum(signalArray) * 1)
        signal_len = len(self.lenkradwin)
        lenkradwin = self.lenkradwin[signal_len-(5*self.global_param['freq']):signal_len-(3*self.global_param['freq'])]
        lenkradwin_mean = np.mean(lenkradwin)
        mean_dH = 0

        if lenkscurve > 0:
            if lenkradwin_mean > 1000 or lenkradwin_mean < 0:
                mean_dH = 0
            elif 100 < lenkradwin_mean < 300:
                mean_dH = 100
            elif 300 <= lenkradwin_mean <= 1000:
                mean_dH = np.interp(lenkradwin_mean, [300, 1000], [100, 0])
            elif 0 <= lenkradwin_mean <= 100:
                mean_dH = np.interp(lenkradwin_mean, [0, 100], [0, 100])

        else:
            if lenkradwin_mean < -1000 or lenkradwin_mean > 0:
                mean_dH = 0
            elif -100 > lenkradwin_mean > -300:
                mean_dH = 100
            elif -100 <= lenkradwin_mean <= 0:
                mean_dH = np.interp(lenkradwin_mean, [-100, 0], [100, 0])
            elif -1000 <= lenkradwin_mean <= -300:
                mean_dH = np.interp(lenkradwin_mean, [-1000, -300], [0, 100])
        return mean_dH

    def compute_module_dH(self):
        self.modul_dH = 0.7 * self.fit_curve_dH() + 0.3 * self.compute_mean_dH()
        return self.modul_dH

# functions to compute evaluation for time duration
class modul_t:
    def __init__(self, lenkradwin, config):
        self.global_param = config #config()
        self.modul_t = 0
        self.lenkradwin = lenkradwin

    def compute_module_t(self):
        meas_t = len(self.lenkradwin)/self.global_param['freq'] # in sec
        if meas_t >= self.global_param['goal_t']:
            self.modul_t = 100
        else:
            self.modul_t = np.interp(meas_t, [0, self.global_param['goal_t']], [0, 100])
        return self.modul_t

# visualisation of some intermediate results
class plotting:
    def __init__(self, modul_R, modul_dH, axs_dH, triggers, limits, velocity):
        self.modul_R_param = modul_R
        self.modul_dH_param = modul_dH
        self.axs = axs_dH
        self.triggers = triggers
        self.limits = limits
        self.Fahrge_DIS = velocity

    def radius_eval(self):
        return self.modul_R_param.radius, self.modul_R_param.avg_radius, \
        self.modul_R_param.R_low_limit, self.modul_R_param.R_upper_limit, \
        self.triggers, self.limits
    
    def extra_measurements(self):
        return self.Fahrge_DIS #velocity in km/h
    
    def plot_curve_dH(self, clr_scatter, marker_scatter, clr_plot):
        #fig, axs = plt.subplots(2)
        data_time = self.modul_dH_param.meas_time
        data_lenkradwin = self.modul_dH_param.avg_lenkradwin.values
        for idx in range(0, len(self.modul_dH_param.meas_time), 75):
            # axs[0].scatter(data_time[idx], data_lenkradwin[idx], color=clr_scatter, marker=marker_scatter)
            plt.scatter(data_time[idx], data_lenkradwin[idx], color=clr_scatter, marker=marker_scatter)
        #axs[0].plot(data_time, self.modul_dH_param.fit_curve, color=clr_plot)
        plt.plot(data_time, self.modul_dH_param.fit_curve, color=clr_plot)
        plt.grid(True)
        plt.show()
