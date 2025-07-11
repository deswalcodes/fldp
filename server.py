import flwr as fl
from flwr.server.strategy import FedAvg

class FedAvgWithMetrics(FedAvg):
    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Collect client accuracies correctly
        accuracies = []
        for _, evaluate_res in results:
            if evaluate_res.metrics and "accuracy" in evaluate_res.metrics:
                accuracies.append(evaluate_res.metrics["accuracy"])
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        print(f"[Server] Round {server_round} average client accuracy: {avg_accuracy:.4f}")

        return aggregated_loss, aggregated_metrics

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=FedAvgWithMetrics(),
)