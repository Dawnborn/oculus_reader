import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import threading
import time

class LivePlotApp:
    def __init__(self, vector_length=3, max_points=100, dimname_list=['rx','ry','rz'], timestep=1):
        self.vector_length = vector_length
        self.dimname_list = dimname_list
        self.timestep=timestep # second
        self.max_points = max_points
        self.x_data = list(range(max_points))
        self.y_data = [np.zeros(max_points).tolist() for _ in range(vector_length)]
        self.lock = threading.Lock()

        self.app = dash.Dash(__name__)
        self.app.layout = html.Div(
            [
                dcc.Graph(id='live-graph'),
                dcc.Interval(
                    id='graph-update',
                    interval=self.timestep*1000,  # in milliseconds
                    n_intervals=0
                )
            ]
        )

        self.app.callback(Output('live-graph', 'figure'),
                          [Input('graph-update', 'n_intervals')])(self.update_graph_live)

        self.thread = threading.Thread(target=self.run_app)
        self.thread.daemon = True
        self.thread.start()

    def run_app(self):
        self.app.run_server(debug=False, use_reloader=False)

    def update(self, vector):
        """
        Update the plot with a new vector.
        
        Parameters:
        vector (list or np.array): An array representing the new data point.
        
        Returns:
        None
        """
        if len(vector) != self.vector_length:
            raise ValueError(f"Expected a vector of length {self.vector_length}")

        with self.lock:
            if len(self.x_data) >= self.max_points:
                self.x_data.pop(0)
                for y in self.y_data:
                    y.pop(0)

            self.x_data.append(self.x_data[-1] + 1 if self.x_data else 0)
            for i in range(self.vector_length):
                self.y_data[i].append(vector[i])

    def update_graph_live(self, n_intervals):
        with self.lock:
            data = []
            for i in range(self.vector_length):
                trace = go.Scatter(
                    x=list(self.x_data),
                    y=list(self.y_data[i]),
                    mode='lines',
                    # name=f'Dimension {i+1}',
                    name = self.dimname_list[i]
                )
                data.append(trace)

            return {'data': data,
                    'layout': go.Layout(xaxis=dict(range=[min(self.x_data), max(self.x_data)]),
                                        yaxis=dict(range=[-3, 3]))}

if __name__ == "__main__":
    timestep=0.5
    live_plot = LivePlotApp(vector_length=3, max_points=100,timestep=timestep)
    while True:
        data = np.random.rand(3)
        live_plot.update(data)
        time.sleep(timestep)