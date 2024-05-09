import json
import matplotlib.pyplot as plt
import re


def read_file():
    file_path = ["./logs/CNN-kernel32.json", "./logs/CNN-kernel64.json", "./logs/CNN-kernel128.json",]
    exp_data = {}
    for path in file_path:
        with open(path, "r") as f:
            data = json.load(f)
            model_name = re.search(r"logs/(.*).json", path).group(1)
            exp_data[model_name] = data
    return exp_data


def loss(exp_data):
    # 根据train_loss和valid_loss绘制loss曲线
    plt.figure()
    for model_name, data in exp_data.items():
        plt.plot(data["train_loss"], label=f"{model_name}_train_loss")
        plt.plot(data["valid_loss"], label=f"{model_name}_valid_loss", linestyle='--')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig("./results/loss_curve.png")


def acc(exp_data):
    # 根据train_acc和valid_acc绘制acc曲线
    plt.figure()
    for model_name, data in exp_data.items():
        plt.plot(data["train_acc"], label=f"{model_name}_train_acc")
        plt.plot(data["valid_acc"], label=f"{model_name}_valid_acc", linestyle='--')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.savefig("./results/acc_curve_kernel.png")


def f1(exp_data):
    # 根据train_f1和valid_f1绘制f1曲线
    plt.figure()
    for model_name, data in exp_data.items():
        plt.plot(data["train_f1"], label=f"{model_name}_train_f1")
        plt.plot(data["valid_f1"], label=f"{model_name}_valid_f1", linestyle='--')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1 Curve")
    plt.savefig("./results/f1_curve.png")


def test(exp_data):
    # 绘制训练时间
    times = [data["time"] for data in exp_data.values()]

    # 假设已经有了一个画布和两个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # 绘制测试准确率
    ax1.bar(exp_data.keys(), [data["test_acc"] for data in exp_data.values()], label="Test Accuracy", color='b',
            width=0.4)
    ax1.set_ylabel("Accuracy", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlabel("Model")
    ax1.set_title("Test Accuracy")
    ax1.legend()

    # 绘制测试F1分数
    ax2.bar(exp_data.keys(), [data["test_f1"] for data in exp_data.values()], label="Test F1", color='r', width=0.4)
    ax2.set_ylabel("F1 Score", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_xlabel("Model")
    ax2.set_title("Test F1 Score")
    ax2.legend()

    # 绘制训练时间
    ax3.bar(exp_data.keys(), times, label="Training Time", color='g', width=0.4)
    ax3.set_ylabel("Time (s)", color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.set_xlabel("Model")
    ax3.set_title("Training Time")
    ax3.legend()

    # 调整子图间距
    plt.tight_layout()

    # 保存图像
    plt.savefig("./results/test_result_kernel.png")


def main():
    exp_data = read_file()
    # print(exp_data)
    # loss(exp_data)
    acc(exp_data)
    # f1(exp_data)
    test(exp_data)


if __name__ == "__main__":
    main()
