import sys

from megatron.training.arguments import core_transformer_config_from_args, parse_args


def test_norm_router_init_method_default_from_cli_is_monte_carlo(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_norm_router_args.py"])

    args = parse_args(ignore_unknown_args=True)

    assert args.moe_norm_routing_init_method == "monte_carlo"


def test_norm_router_init_method_default_propagates_to_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_norm_router_args.py", "--moe-norm-routing"])

    args = parse_args(ignore_unknown_args=True)
    config = core_transformer_config_from_args(args)

    assert config.moe_norm_routing is True
    assert config.moe_norm_routing_init_method == "monte_carlo"
