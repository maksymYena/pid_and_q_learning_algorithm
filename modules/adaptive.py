# modules/adaptive.py

import numpy as np

class PIDAdaptiveController:
    def __init__(self, cfg):
        # Только сохраняем начальные PID-параметры
        self.params = np.array([cfg['pid']['Kp'], cfg['pid']['Ki'], cfg['pid']['Kd']])

    def reset(self):
        # Если нужно — вернуть к исходным (в cfg)
        pass

    def get_params(self):
        # Возвращает текущие [Kp, Ki, Kd]
        return self.params

    def apply_action(self, action):
        # action — дельты (dKp, dKi, dKd)
        self.params += action

    def compute_control(self, error, dt):
        # простой PID на базе self.params
        Kp, Ki, Kd = self.params
        # храним интеграл и prev_error как атрибуты
        if not hasattr(self, 'integral'):
            self.integral = 0.0
            self.prev_error = 0.0
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return Kp * error + Ki * self.integral + Kd * derivative
