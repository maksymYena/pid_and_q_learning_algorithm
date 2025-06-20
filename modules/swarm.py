# modules/swarm.py

import numpy as np

class SwarmController:
    def __init__(self, cfg):
        self.eta = cfg['eta']

    def step(self, positions, velocities, paths, dt):
        """
        positions: np.ndarray shape (N, 2)
        velocities: np.ndarray shape (N, 2)
        paths: list of N списков координат пути [[x1,y1], [x2,y2], ...]
        dt: шаг времени
        """
        N = positions.shape[0]
        new_positions = np.zeros_like(positions)
        new_velocities = np.zeros_like(velocities)

        for i in range(N):
            pos = positions[i]
            vel = velocities[i]
            # Соседи — все остальные агенты
            neighbors_pos = np.delete(positions, i, axis=0)
            neighbors_vel = np.delete(velocities, i, axis=0)

            # Правила Boids
            v_sep = self._sep(pos, neighbors_pos)
            v_coh = self._coh(pos, neighbors_pos)
            v_ali = self._ali(vel, neighbors_vel)

            # Движение по маршруту
            path = paths[i]
            # убеждаемся, что в пути есть хотя бы 2 точки
            if len(path) >= 2:
                goal_vec = np.array(path[1]) - pos
            else:
                goal_vec = np.zeros_like(pos)

            # Обновляем скорость и позицию
            new_vel = vel + self.eta * (v_sep + v_coh + v_ali) + goal_vec * dt
            new_pos = pos + new_vel * dt

            new_velocities[i] = new_vel
            new_positions[i] = new_pos

        return new_positions, new_velocities

    def _sep(self, pos, neighbors):
        """Разделение: избегаем столкновений."""
        if len(neighbors) == 0:
            return np.zeros_like(pos)
        vec = np.zeros_like(pos)
        for p_j in neighbors:
            diff = pos - p_j
            dist2 = np.dot(diff, diff) + 1e-6
            vec += diff / dist2
        return vec

    def _coh(self, pos, neighbors):
        """Когезия: стремимся к центру масс."""
        if len(neighbors) == 0:
            return np.zeros_like(pos)
        center = np.mean(neighbors, axis=0)
        return center - pos

    def _ali(self, vel, neighbors_vel):
        """Выравнивание: стремимся к средней скорости."""
        if len(neighbors_vel) == 0:
            return np.zeros_like(vel)
        avg_vel = np.mean(neighbors_vel, axis=0)
        return avg_vel - vel
