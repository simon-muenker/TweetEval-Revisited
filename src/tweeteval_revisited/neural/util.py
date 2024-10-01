import torch


def get_model_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_model_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_memory_usage(model: torch.nn.Module) -> str:
    usage_in_byte: int = sum(
        [
            sum(
                [
                    param.nelement() * param.element_size()
                    for param in model.parameters()
                ]
            ),
            sum([buf.nelement() * buf.element_size() for buf in model.buffers()]),
        ]
    )

    return f"{usage_in_byte / (1024.0 * 1024.0):2.4f} MB"
