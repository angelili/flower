"""Flower server example."""


import flwr as fl

if __name__ == "__main__":
    fl.server.start_server(
        server_address="10.10.21.21:8080",
        config=fl.server.ServerConfig(num_rounds=3),
    )
