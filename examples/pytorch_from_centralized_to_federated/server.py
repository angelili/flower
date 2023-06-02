"""Flower server example."""


import flwr as fl

if __name__ == "__main__":
    fl.server.start_server(
        server_address="10.30.0.8:6379",
        config=fl.server.ServerConfig(num_rounds=3),
    )
