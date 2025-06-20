# modules/planner.py

import networkx as nx

class GraphPlanner:
    def __init__(self, cfg):
        """
        cfg ожидает в себе разделы:
          cfg['graph'], cfg['traffic'], cfg['weather'], cfg['planner']['lambda'], cfg['planner']['mu']
        """
        self.lambda_ = cfg['planner']['lambda']
        self.mu = cfg['planner']['mu']

        # Собираем граф из cfg['graph']
        self.graph = nx.DiGraph()
        for u, nbrs in cfg['graph'].items():
            for v, d in nbrs:
                self.graph.add_edge(u, v, weight=d)

        # Сохраняем traffic & weather
        self.traffic = {
            tuple(edge.split('-')): val
            for edge, val in cfg['traffic'].items()
        }
        self.weather = {
            tuple(edge.split('-')): val
            for edge, val in cfg['weather'].items()
        }

    def _update_edge_costs(self):
        # Пересчитываем атрибут 'cost' для каждого ребра
        for u, v, data in self.graph.edges(data=True):
            d = data['weight']
            rho = self.traffic.get((u, v), 0.0)
            wthr = self.weather.get((u, v), 0.0)
            self.graph[u][v]['cost'] = d * (1 + self.lambda_ * rho + self.mu * wthr)

    def compute_path(self, start, goal):
        """
        Возвращает список узлов кратчайшего пути по динамическим весам 'cost'.
        """
        self._update_edge_costs()
        return nx.shortest_path(self.graph, source=start, target=goal, weight='cost')
