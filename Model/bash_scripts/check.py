import torch
import sys

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    if device.type != 'cuda':
        print('No GPU available')
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
