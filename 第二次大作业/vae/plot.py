import matplotlib.pyplot as plt

# Path to the log file
log_file_path = './logs/2024_01_09_0652_2_training.log'  # Replace with your log file path

def parse_log_file(log_file_path):
    epochs = []
    train_losses = []
    val_losses = []

    with open(log_file_path, 'r') as file:
        log_content = file.readlines()

    for line in log_content:
        parts = line.strip().split(',')
        epoch = int(parts[1].split(' ')[1])
        train_loss = float(parts[2].split(' ')[3])
        val_loss = float(parts[3].split(' ')[3])

        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return epochs, train_losses, val_losses

# Plotting the training and validation loss
def plot_losses(epochs, train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('./loss_fig/2.png')

epochs, train_losses, val_losses = parse_log_file(log_file_path)
plot_losses(epochs, train_losses, val_losses)
