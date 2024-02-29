import torch

def calculate_accuracy(pred_labels, labels):
    # _, predicted = torch.max(outputs.data, 1)
    # total = labels.size(0)
    # correct = (predicted == labels).sum().item()
    # return 100 * correct / total
    acc=0.0
    correct = 0.0
    for p, g in zip(pred_labels, labels):
        print("p: ", p)
        print("g: ", g)
        if p == g:
            correct += 1
    acc = 100 * correct/len(pred_labels)
    print("Accuracy for batch ", acc)
    return acc