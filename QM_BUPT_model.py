import numpy as np
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class SquareRootFitter:
    def __init__(self, x_data, y_data):
        self.x_data = list(x_data)
        self.y_data = list(y_data)
        self._fit_model()

    def _model(self, x, a, b):
        return a * np.sqrt(x) + b

    def _fit_model(self):
        x = np.array(self.x_data)
        y = np.array(self.y_data)
        self.params, _ = curve_fit(self._model, x, y)
        self.a, self.b = self.params

    def predict(self, x):
        if isinstance(x, list):
            return [round(self._model(xi, self.a, self.b)) for xi in x]
        else:
            return round(self._model(x, self.a, self.b))

    def add_data_point(self, x_new, y_new):
        self.x_data.append(x_new)
        self.y_data.append(y_new)
        self._fit_model()

    def show_fit(self):
        x = np.array(self.x_data)
        y = np.array(self.y_data)

        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = self._model(x_fit, self.a, self.b)

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, label='Data Points', color='blue')
        plt.plot(x_fit, y_fit, label=f'Fitted: y = {self.a:.3f}√x + {self.b:.3f}', color='red')
        plt.xlabel('Input (x)')
        plt.ylabel('Output (y)')
        plt.title('Square Root Fit Model')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_equation(self):
        return f"y = round({self.a:.3f} * sqrt(x) + {self.b:.3f})"
