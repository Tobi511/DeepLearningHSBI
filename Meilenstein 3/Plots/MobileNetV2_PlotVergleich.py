import matplotlib.pyplot as plt

def read_training_log(filepath):
    acc, val_acc, loss, val_loss = [], [], [], []
    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        acc.append(float(parts[1]))
        val_acc.append(float(parts[2]))
        loss.append(float(parts[3]))
        val_loss.append(float(parts[4]))
    return acc, val_acc, loss, val_loss

# === Dateipfade ===
logfile1 = ("../Fine Tuning/results/results_EfficientNetB0_tuning/results_EfficientNetB0_tuning_r1"
            "/transfertrainingsverlauf_EfficientNetB0_tuning_r1.txt")
logfile2 = ("../Feature Extraction/results/results_EfficientNetB0_FE/results_EfficientNetB0_FE_r1"
            "/transfertrainingsverlauf_EfficientNetB0_FE_r1.txt")
plot_path = "EfficientNetB0_vergleich.png"

# === Einlesen ===
acc1, val_acc1, loss1, val_loss1 = read_training_log(logfile1)
acc2, val_acc2, loss2, val_loss2 = read_training_log(logfile2)
epochs1 = list(range(1, len(acc1) + 1))
epochs2 = list(range(1, len(acc2) + 1))

# === Gemeinsame Achsgrenzen berechnen ===
acc_all = acc1 + val_acc1 + acc2 + val_acc2
loss_all = loss1 + val_loss1 + loss2 + val_loss2
acc_min, acc_max = min(acc_all), max(acc_all)
loss_min, loss_max = min(loss_all), max(loss_all)

# Etwas Puffer geben
acc_ylim = (max(0.0, acc_min - 0.02), min(1.0, acc_max + 0.02))
loss_ylim = (0.0, loss_max + 0.1)

# === Plot ===
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Logdatei 2
axs[0, 0].plot(epochs2, acc2, label='Feature Extraction', linestyle='-', marker='o')
axs[0, 0].plot(epochs1, acc1, label='Fine Tuning', linestyle='--', marker='x')
axs[0, 0].set_title("Train Accuracy")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Accuracy")
axs[0, 0].grid(True, linestyle='--', linewidth=0.5)
axs[0, 0].legend()
axs[0, 0].set_ylim(0.98, 1.0)

axs[0, 1].plot(epochs2, loss2, label='Feature Extraction', linestyle='-', marker='o')
axs[0, 1].plot(epochs1, loss1, label='Fine Tuning', linestyle='--', marker='x')
axs[0, 1].set_title("Train Loss")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Loss")
axs[0, 1].grid(True, linestyle='--', linewidth=0.5)
axs[0, 1].legend()
axs[0, 1].set_ylim(0.0, 0.01)


axs[1, 0].plot(epochs2, val_acc2, label='Feature Extraction', linestyle='-', marker='o')
axs[1, 0].plot(epochs1, val_acc1, label='Fine Tuning', linestyle='--', marker='x')
axs[1, 0].set_title("Validation Accuracy")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Accuracy")
axs[1, 0].grid(True, linestyle='--', linewidth=0.5)
axs[1, 0].legend()
axs[1, 0].set_ylim(0.98, 1.0)

axs[1, 1].plot(epochs2, val_loss2, label='Feature Extraction', linestyle='-', marker='o')
axs[1, 1].plot(epochs1, val_loss1, label='Fine Tuning', linestyle='--', marker='x')
axs[1, 1].set_title("Validation Loss")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Loss")
axs[1, 1].grid(True, linestyle='--', linewidth=0.5)
axs[1, 1].legend()
axs[1, 1].set_ylim(0.0, 0.01)



#plt.tight_layout()

fig.suptitle("EfficientNetB0: Feature Extraction vs. Fine Tuning", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(plot_path)
