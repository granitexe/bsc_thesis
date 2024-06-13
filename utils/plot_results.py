import matplotlib.pyplot as plt

def plot_training_results(training_history, title="Training Loss"):
    plt.figure(figsize=(10, 5))
    plt.plot(training_history['loss'], label='Training Loss')
    plt.plot(training_history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_evaluation_results(results, title="Evaluation Results"):
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), results.values())
    plt.title(title)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.show()
