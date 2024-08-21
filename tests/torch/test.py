import torch
import custom_op


if __name__ == "__main__":
    tensor = torch.randn(10, device = 'cuda')

    print("Before custom op: ", tensor)
    custom_op.add_one(tensor)

    print("After custom op: ", tensor)