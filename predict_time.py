import torch
import torch.nn as nn
# include whatever other imports you need here


class TimePredictionNetwork(nn.Module):
    # Your network definition goes here
    pass


def predict(images):
    # Determine which device the input tensor is on
    device = torch.device("cuda" if images.is_cuda else "cpu")

    # Add your model init parameters here if you have any
    model = TimePredictionNetwork()
    # Move to same device as input images
    model = model.to(device)
    # Load network weights
    model.load_state_dict(torch.load(
        'weights.pkl', map_location=torch.device(device)))
    # Put model in evaluation mode
    model.eval()

    # Optional: do whatever preprocessing you do on the images
    # if not included as tranformations inside the model

    with torch.no_grad():
        # Pass images to model
        predicted_times = model(images)

    # If your output needs any post-processing, do it here

    # Return predicted times
    return predicted_times
