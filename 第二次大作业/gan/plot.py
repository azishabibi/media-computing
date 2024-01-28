import matplotlib.pyplot as plt
import re

def plot_losses_from_log(log_file_path):
    with open(log_file_path, 'r') as file:
        log_lines = file.readlines()
    epochs = []
    d_losses = []
    g_losses = []

    # Regular expression to match the relevant information in each line
    pattern = re.compile(r'Epoch: (\d+), D Loss: ([\d.]+), G Loss: ([\d.]+)')

    # Extract data from each line
    for line in log_lines:
        match = pattern.search(line)
        if match:
            epoch, d_loss, g_loss = match.groups()
            epochs.append(int(epoch))
            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, d_losses, label='D Loss')
    plt.plot(epochs, g_losses, label='G Loss')
    plt.title('D Loss and G Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./loss/loss.png')

# Path to the log file, change it for your file
log_file_path = './logs/2024_01_10_0202_training.log'

plot_losses_from_log(log_file_path)
