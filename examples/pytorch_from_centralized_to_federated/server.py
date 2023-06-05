"""Flower server example."""
fedl_server="10.30.0.254:9000"
fedl_no_proxy=True
import flwr as fl



if __name__ == "__main__":
    if fedl_no_proxy:
      import os
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    fl.server.start_server(
        server_address= "10.30.0.254:9000",
        config=fl.server.ServerConfig(num_rounds=3),
    )
