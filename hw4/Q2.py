from torchvision import models, transforms
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
from captum.attr import Saliency, InputXGradient, IntegratedGradients
import pickle
from PIL import Image
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def explainer(model, labels_human, DEVICE):
    model.to(DEVICE)
    model.eval()

    # Explainer 
    attribution = Saliency(model)

    # Load images and labels
    with open("sample_imagenetdata", 'rb') as f:
        image_data = pickle.load(f)
    fig, ax = plt.subplots(5,3, figsize=(30,50))
    i=0
    for key,value in image_data.items():
        X = value['image']
        y = value['label']
        label = value['label_human']
        X, y = X.to(DEVICE), y.to(DEVICE)
        # Predict the label
        output = model(X)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_label = torch.max(probabilities, 0)[1].item()

        # Compute the attribution scores using Saliency for true label
        attr_true = attribution.attribute(inputs=X, target=y)

        # Compute the attribution scores using Saliency for predicted label
        attr_pred = attribution.attribute(inputs=X, target=predicted_label)

        # Transform the image to original scale
        X = X * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        
        # Visualize the attribution scores for true label
        explainer_true, _ = torch.max(attr_true.data.abs(), dim=1) 
        explainer_true = explainer_true.cpu().detach().numpy()
        explainer_true = (explainer_true-explainer_true.min())/(explainer_true.max()-explainer_true.min())

        # Visualize the attribution scores for predicted label
        explainer_pred, _ = torch.max(attr_pred.data.abs(), dim=1)
        explainer_pred = explainer_pred.cpu().detach().numpy()
        explainer_pred = (explainer_pred-explainer_pred.min())/(explainer_pred.max()-explainer_pred.min())
        ax[i][0].imshow(X[0].permute(1, 2, 0).to('cpu'))
        ax[i][1].imshow(explainer_true[0])
        ax[i][1].set_title(f"True: {label[0]}", fontsize=48)
        ax[i][2].imshow(explainer_pred[0])
        ax[i][2].set_title(f"Predicted: {labels_human[predicted_label][0]}", fontsize=48)
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        ax[i][1].set_xticks([])
        ax[i][1].set_yticks([])
        ax[i][2].set_xticks([])
        ax[i][2].set_yticks([])
        i+=1
    fig.subplots_adjust(wspace=0, hspace=0, top=1.0)
    plt.savefig("Saliency.png", bbox_inches='tight')

def part3_custom_images(model, labels_human, DEVICE):
    model.to(DEVICE)
    model.eval()
    
    attribution = Saliency(model)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    image_paths = ['assets/goldfish.jpg', 'assets/hummingbird.jpg', 'assets/black_swan.jpg', 'assets/golden_retriever.jpg', 'assets/daisy.jpg']
    ground_truth_labels = [1, 94, 100, 207, 985]
    
    fig, ax = plt.subplots(5, 3, figsize=(30, 50))
    
    for i, (img_path, gt_label) in enumerate(zip(image_paths, ground_truth_labels)):
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        img_normalized = transform_normalize(img_tensor)
        img_normalized = img_normalized.unsqueeze(0).to(DEVICE)
        
        output = model(img_normalized)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_label = torch.max(probabilities, 0)[1].item()
        
        attr_true = attribution.attribute(inputs=img_normalized, target=gt_label)
        attr_pred = attribution.attribute(inputs=img_normalized, target=predicted_label)
        
        img_denorm = img_normalized * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
        
        explainer_true, _ = torch.max(attr_true.data.abs(), dim=1)
        explainer_true = explainer_true.cpu().detach().numpy()
        explainer_true = (explainer_true - explainer_true.min()) / (explainer_true.max() - explainer_true.min())
        
        explainer_pred, _ = torch.max(attr_pred.data.abs(), dim=1)
        explainer_pred = explainer_pred.cpu().detach().numpy()
        explainer_pred = (explainer_pred - explainer_pred.min()) / (explainer_pred.max() - explainer_pred.min())
        
        ax[i][0].imshow(img_denorm[0].permute(1, 2, 0).cpu().clamp(0, 1))
        ax[i][1].imshow(explainer_true[0])
        ax[i][1].set_title(f"True: {labels_human[gt_label][0]}", fontsize=48)
        ax[i][2].imshow(explainer_pred[0])
        ax[i][2].set_title(f"Predicted: {labels_human[predicted_label][0]}", fontsize=48)
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        ax[i][1].set_xticks([])
        ax[i][1].set_yticks([])
        ax[i][2].set_xticks([])
        ax[i][2].set_yticks([])
    
    fig.subplots_adjust(wspace=0, hspace=0, top=1.0)
    plt.savefig("Part3_Saliency.png", bbox_inches='tight')

def part4_additional_methods(model, labels_human, DEVICE):
    model.to(DEVICE)
    model.eval()
    
    inputxgrad = InputXGradient(model)
    ig = IntegratedGradients(model)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    image_paths = ['assets/goldfish.jpg', 'assets/hummingbird.jpg', 'assets/black_swan.jpg', 'assets/golden_retriever.jpg', 'assets/daisy.jpg']
    ground_truth_labels = [1, 94, 100, 207, 985]
    
    fig, ax = plt.subplots(5, 5, figsize=(50, 50))
    
    for i, (img_path, gt_label) in enumerate(zip(image_paths, ground_truth_labels)):
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        img_normalized = transform_normalize(img_tensor)
        img_normalized = img_normalized.unsqueeze(0).to(DEVICE)
        
        output = model(img_normalized)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_label = torch.max(probabilities, 0)[1].item()
        
        attr_ixg_true = inputxgrad.attribute(inputs=img_normalized, target=gt_label)
        attr_ixg_pred = inputxgrad.attribute(inputs=img_normalized, target=predicted_label)
        attr_ig_true = ig.attribute(inputs=img_normalized, target=gt_label)
        attr_ig_pred = ig.attribute(inputs=img_normalized, target=predicted_label)
        
        img_denorm = img_normalized * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
        
        explainer_ixg_true, _ = torch.max(attr_ixg_true.data.abs(), dim=1)
        explainer_ixg_true = explainer_ixg_true.cpu().detach().numpy()
        explainer_ixg_true = (explainer_ixg_true - explainer_ixg_true.min()) / (explainer_ixg_true.max() - explainer_ixg_true.min())
        
        explainer_ixg_pred, _ = torch.max(attr_ixg_pred.data.abs(), dim=1)
        explainer_ixg_pred = explainer_ixg_pred.cpu().detach().numpy()
        explainer_ixg_pred = (explainer_ixg_pred - explainer_ixg_pred.min()) / (explainer_ixg_pred.max() - explainer_ixg_pred.min())
        
        explainer_ig_true, _ = torch.max(attr_ig_true.data.abs(), dim=1)
        explainer_ig_true = explainer_ig_true.cpu().detach().numpy()
        explainer_ig_true = (explainer_ig_true - explainer_ig_true.min()) / (explainer_ig_true.max() - explainer_ig_true.min())
        
        explainer_ig_pred, _ = torch.max(attr_ig_pred.data.abs(), dim=1)
        explainer_ig_pred = explainer_ig_pred.cpu().detach().numpy()
        explainer_ig_pred = (explainer_ig_pred - explainer_ig_pred.min()) / (explainer_ig_pred.max() - explainer_ig_pred.min())
        
        ax[i][0].imshow(img_denorm[0].permute(1, 2, 0).cpu().clamp(0, 1))
        ax[i][0].set_title(f"Original", fontsize=48, pad=80)
        ax[i][1].imshow(explainer_ixg_true[0])
        ax[i][1].set_title(f"InputXGrad True: {labels_human[gt_label][0]}", fontsize=32, pad=80)
        ax[i][2].imshow(explainer_ixg_pred[0])
        ax[i][2].set_title(f"InputXGrad Pred: {labels_human[predicted_label][0]}", fontsize=32, pad=80)
        ax[i][3].imshow(explainer_ig_true[0])
        ax[i][3].set_title(f"IntegratedGrad True: {labels_human[gt_label][0]}", fontsize=32, pad=80)
        ax[i][4].imshow(explainer_ig_pred[0])
        ax[i][4].set_title(f"IntegratedGrad Pred: {labels_human[predicted_label][0]}", fontsize=32, pad=80)
        
        for j in range(5):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    
    fig.subplots_adjust(wspace=1, hspace=0.3, top=1.0)
    plt.savefig("Part4_Additional_Methods.png", bbox_inches='tight')

if __name__=="__main__":

    # Create the model
    model_googlenet= models.googlenet(pretrained=True)

    # Summary of the model
    print(summary(model=model_googlenet, input_size=(1, 3, 224, 224), col_width=20, col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'], verbose=0))
    
    # Load classes to human readable labels
    labels_human = {}
    with open(f'imagenet1000_clsidx_to_labels.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace("'", "").strip(",")
            if "{" in line or "}" in line:
                continue
            else:
                idx = int(line.split(":")[0])
                lbl = line.split(":")[1].split(",")
                labels_human[idx] = [x.strip() for x in lbl]
    # Explainer
    # explainer(model_googlenet, labels_human,  DEVICE)
    
    # part3_custom_images(model_googlenet, labels_human, DEVICE)

    part4_additional_methods(model_googlenet, labels_human, DEVICE)