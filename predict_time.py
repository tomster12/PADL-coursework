import torch
import torch.nn as nn
import torchvision.io as torchio


def preprocess_images(images):
    # Input images is (Bx3x448x448) in float range [0..1]
    # Resize to (Bx3x224x224) with interpolation
    images = torch.nn.functional.interpolate(images, size=224, mode="bilinear")
    # Greyscale to (Bx1x224x224) with mean
    images = torch.mean(images, dim=1, keepdim=True)
    return images


def onehot_to_label(onehot):
    # Input onehot is (Bx720)
    # Convert to label (Bx2)
    label = torch.argmax(onehot, dim=1)
    return torch.stack([label // 60, label % 60], dim=1)


# CNN for clock face classification
# - Input is (Bx1x224x224) greyscale images
# - Output is (Bx720) classes for each minute of the day
class ClockClassificationCNN(nn.Module):
    def __init__(self):
        super(ClockClassificationCNN, self).__init__()
        self.net = nn.Sequential(
            # Bx1x224x224 -> Bx16x112x112
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Bx16x112x112 -> Bx32x56x56
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Bx32x56x56 -> Bx48x28x28
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            # Bx(48x28x28) -> Bx(12x60)
            nn.Linear(48 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 12 * 60)
        )

    def forward(self, x):
        return self.net(x)


def predict(images):
    # Images come in as (Bx3x448x448) in range [0..1]
    # Preprocess images to (Bx1x224x224)
    images = preprocess_images(images)

    # Use GPU if available
    device = torch.device("cuda" if images.is_cuda else "cpu")

    # Load model and weights
    model = ClockClassificationCNN()
    model = model.to(device)
    model.load_state_dict(torch.load(
        "clock_classifer_model.pt", map_location=torch.device(device)))

    # Predict time with model
    model.eval()
    with torch.no_grad():
        predicted_times = model(images)

    # Convert output to label
    predicted_times = onehot_to_label(predicted_times)

    # Return predicted times (Bx2)
    return predicted_times


if __name__ == "__main__":
    # Load 1 image and label from dataset
    images = torchio.read_image("clocks_dataset/train/0000.png")
    real_time = open("clocks_dataset/train/0000.txt", "r").read()

    # Predict time with (Bx3x448x448) images in range [0..1]
    predicted_time = predict(images.unsqueeze(0) / 255.0)
    print(predicted_time.shape)
    print(predicted_time)
    print(real_time)
