import os
class Greeter:
    def train_model(self):
        directory = os.path.join(os.path.dirname(__file__), 'data')
        print(f'Directory path: {directory}')
