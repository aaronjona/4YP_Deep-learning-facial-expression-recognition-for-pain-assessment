import glob
import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import guided_r3d
import guided_r2p1d
import r2p1d
import r3d


class GuidedReLU(nn.Module):
    """
    Custom ReLU for Guided Backpropagation that only passes positive gradients 
    for positive inputs.
    """
    def forward(self, input):
        return F.relu(input, inplace=False)
    
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[grad_output < 0] = 0
        return grad_input
    
def replace_relu_with_guided(model):
    """
    Replace all instances of nn.ReLU with GuidedReLU in the model.
    """
    relu_modules = []
    # First, collect all the names and modules of ReLU
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            relu_modules.append((name, module))

    # Then, replace each ReLU with GuidedReLU
    for name, _ in relu_modules:
        # Check if the ReLU module has a parent or is top-level
        if '.' in name:
            parent_module, relu_name = name.rsplit('.', 1)
            parent = dict(model.named_modules())[parent_module]  # Get the parent module using its path
            setattr(parent, relu_name, GuidedReLU())
        else:
            # The ReLU module is a direct attribute of the model
            setattr(model, name, GuidedReLU())




def get_guided_backprop(model, input_tensor):
    """
    Function to compute guided backpropagation gradients.
    """
    replace_relu_with_guided(model)
    input_tensor.requires_grad = True
    
    # Forward pass
    output = model(input_tensor)
    
    # Zero gradients
    model.zero_grad()
    
    # Target for backprop
    idx = output.argmax(dim=1).item()
    target = output[0, idx]
    
    # Backward pass
    target.backward()

    # Return gradients for input image
    return input_tensor.grad

def get_guided_backprop_mask(backprop_gradients, selection_scheme='max'):

    if selection_scheme == 'max':
        guided_backprop_mask, _= torch.max(backprop_gradients.squeeze(), dim=0, keepdim=False)
    elif selection_scheme == 'mean': 
        guided_backprop_mask = torch.mean(backprop_gradients.squeeze(), dim=0)
    elif selection_scheme == 'centre':
        backprop_gradients = backprop_gradients.squeeze()
        guided_backprop_mask = backprop_gradients[backprop_gradients.shape[0] // 2, :, :]
    guided_backprop_mask = guided_backprop_mask.detach().cpu().numpy()
    guided_backprop_mask = (guided_backprop_mask - np.min(guided_backprop_mask)) / (np.max(guided_backprop_mask) - np.min(guided_backprop_mask)) 
    guided_backprop_mask = np.uint8(255 * guided_backprop_mask)  
    return guided_backprop_mask


# def register_hooks(model, model_name):
#     if model_name =='r2p1d':
#         for module in model.modules():
#             if isinstance(module, guided_r2p1d.GuidedReLU):
#                 module.register_backward_hook(module.backward_hook)
#     else:
#         for module in model.modules():
#             if isinstance(module, guided_r3d.GuidedReLU):
#                 module.register_backward_hook(module.backward_hook)

def get_grad_cam(model, input_tensor, target_layer):
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    # Register the hooks
    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_full_backward_hook(backward_hook)

    # Forward Pass
    output = model(input_tensor)
    model.zero_grad()

    idx = output.argmax(dim=1).item()
    target = output[0, idx]
    target.backward()

    # Get Grad-Cam
    gradients_val = gradients['value']  # Shape: [N, C, T, H, W]
    activations_val = activations['value']
    weights = torch.mean(gradients_val, axis=[2, 3, 4], keepdim=True)
    grad_cam = torch.sum(weights * activations_val, axis=1, keepdim=True)
    grad_cam = F.relu(grad_cam)

    # Remove hooks
    hook_forward.remove()
    hook_backward.remove()

    return grad_cam

def get_guided_gradients(model, input_tensor):
    input_tensor.requires_grad_()  # Ensure the tensor requires gradients
    output = model(input_tensor)
    model.zero_grad()
    
    idx = output.argmax(dim=1).item()
    target = output[0, idx]
    target.backward()
    # print(f'guided_backprop shape: {input_tensor.grad.shape}')
    output = input_tensor.grad
    # output = output.squeeze()
    # depth = output.shape[0]
    # output_frame = output[depth // 2, :, :]
    # print(f'guided_backprop output frame shape: {output_frame.shape}')
    return output

def get_heatmap(grad_cam):
    grad_cam_aggregated = torch.mean(grad_cam, dim=1, keepdim=True)  # Shape: [N, 1, D, H, W]
    grad_cam_2d = torch.mean(grad_cam_aggregated, dim=2, keepdim=False)  # Shape: [N, 1, H, W]
    heatmap_2d = grad_cam_2d.squeeze() # Shape: [H, W]
    heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min())  # Normalize
    # print(f'gradcam heatmap shape: {heatmap_2d.shape}')
    heatmap_2d_np = heatmap_2d.cpu().detach().numpy().astype('float32')
    heatmap_resized = cv2.resize(heatmap_2d_np, (256, 256), interpolation = cv2.INTER_LINEAR)
    grad_cam_mask = np.uint8(heatmap_resized * 255)
    # print(f'gradcam heatmap resized shape: {grad_cam_mask.shape}')
    return grad_cam_mask



def get_guided_grad_cam(grad_cam_mask, back_prop_mask):
    guided_grad_cam = (grad_cam_mask.astype(np.float32) / 255) * (back_prop_mask.astype(np.float32) / 255)
    guided_grad_cam = np.uint8(guided_grad_cam * 255)
    return guided_grad_cam


def visualize_guided_grad_cam(input_tensor, guided_grad_cam, count, model_depth, model_name):
    input_tensor = input_tensor.squeeze()# [25, 256, 256]
    depth = input_tensor.shape[0]
    input_image = input_tensor[depth // 2, :, :] # [1, 256, 256]
    input_image = input_image.squeeze().cpu().detach().numpy().astype('float32')  # [256, 256]
    original_image = np.uint8(input_image * 255)
    # print(f'original_image shape: {original_image.shape}')
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    if guided_grad_cam.requires_grad:
        guided_grad_cam = guided_grad_cam.detach()
    if guided_grad_cam.is_cuda:
        guided_grad_cam = guided_grad_cam.cpu()
    guided_grad_cam_np = guided_grad_cam.numpy()
    guided_grad_cam_np = (guided_grad_cam_np - np.min(guided_grad_cam_np)) / (np.max(guided_grad_cam_np) - np.min(guided_grad_cam_np) + 1e-9)  # Normalize
    guided_grad_cam_np = np.uint8(255 * guided_grad_cam_np)
    # print(f'guided_grad_cam_np shape: {guided_grad_cam_np.shape}')
    heatmap_colored = cv2.applyColorMap(guided_grad_cam_np, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_colored, 0.5, original_image_rgb, 0.5, 0)
    
    output_path = f'./outputs_7/output_{model_name}_{model_depth}/heatmap_output_guided/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(output_path + f'guided_grad_cam_{count}.jpg', superimposed_img)
    
    return superimposed_img

def visualize_guided_backprop(guided_gradients, count, model_depth, model_name):
    guided_gradients_np = guided_gradients.cpu().numpy()
    guided_gradients_np = (guided_gradients_np - np.min(guided_gradients_np)) / (np.max(guided_gradients_np) - np.min(guided_gradients_np) + 1e-9)
    gradient_image = np.uint8(255 * guided_gradients_np)

    output_path = f'./output_{model_name}_{model_depth}/guided_backprop_vis/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(output_path + f'guided_backprop_{count}.png', gradient_image)
 

def visualize_guided_grad(guided_gradients, grad_cam, count, model_depth, model_name):
    guided_gradients_np = guided_gradients.cpu().numpy()
    guided_gradients_np = (guided_gradients_np - np.min(guided_gradients_np)) / (np.max(guided_gradients_np) - np.min(guided_gradients_np))
    gradient_image = np.uint8(255 * guided_gradients_np)

    gradient_image_rgb = cv2.cvtColor(gradient_image, cv2.COLOR_GRAY2BGR)
    
    heat_map = np.uint8(255 * grad_cam)
    heatmap_colored = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET) #[256,256,3]

    superimposed_img = cv2.addWeighted(heatmap_colored, 0.5, gradient_image_rgb, 0.5, 0)

    output_path = f'./output_{model_name}_{model_depth}/guided_grad/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(output_path + f'guided_backprop_{count}.png', superimposed_img)

def visualize_rgb(mask):
    coloured_heat_map = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return coloured_heat_map

def save_fig(fig, count, model_depth, model_name, mask_name):
    output_path = f'./outputs_7/output_{model_name}_{model_depth}/rgb/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(output_path + f'{mask_name}_{count}.png', fig)

def save_gray_fig(fig, count, model_depth, model_name, mask_name):
    output_path = f'./outputs_7/output_{model_name}_{model_depth}/gray_scale/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(output_path + f'{mask_name}_{count}.png', fig)



if __name__ == '__main__':
        
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1) 

        # Get Input Tensor for evaluation
        person_path = glob.glob('/data/spet5167/biovid/cropped_tensor_2/*')

        eval_loader = utils.load_eval_data(person_path)

        models = [(int(path.split('_')[-1]), path.split('_')[-2]) for path in glob.glob('./outputs_7/output_*')]
        criterion = nn.CrossEntropyLoss()
        for depth, model_name in models:

                if model_name == 'r2p1d':
                        model =r2p1d.generate_model(model_depth = depth,
                                                    n_classes = 1,
                                                    n_input_channels = 1,
                                                    conv1_t_size = 7,
                                                    conv1_t_stride = 1, 
                                                    no_max_pool = True)
                        target_conv_layer = model.layer4[-1].conv2_s
                elif model_name == 'r3d':
                        model =r3d.generate_model(model_depth = depth,
                                                    n_classes = 1,
                                                    n_input_channels = 1,
                                                    conv1_t_size = 7,
                                                    conv1_t_stride = 1, 
                                                    no_max_pool = True)
                        target_conv_layer = model.layer4[-1].conv2
                else: 
                        continue
        # Instantiate Model
                model = model.cuda() if torch.cuda.is_available() else model
                print(model)

                model.load_state_dict(torch.load(f'./outputs_7/output_{model_name}_{depth}/best_model.pth'))
                model.eval()

                count = 0
                print(f'Model: {model_name}_{depth}:')
                for data, label in eval_loader:
                        data = data.cuda()

                        output = model(data)
                        _, predicted = torch.max(output, 1)

                        print(f'{count} image: label : {label}; predicted :{predicted}.')

                        # Grad-CAM
                        grad_cam = get_grad_cam(model, data, target_conv_layer) # Shape: [N, C, D, H, W]
                        grad_cam_mask = get_heatmap(grad_cam)
                        save_gray_fig(grad_cam_mask, count, depth, model_name, 'Grad_CAM')

                        Grad_heatmap = visualize_rgb(grad_cam_mask)
                        save_fig(Grad_heatmap, count, depth, model_name, 'Grad_CAM')


                        # Guided Backpropagation
                        replace_relu_with_guided(model)

                        guided_gradients = get_guided_backprop(model, data) # shape: [1,1,25,256,256]
                        guided_gradients_mask = get_guided_backprop_mask(guided_gradients)
                        save_gray_fig(guided_gradients_mask, count, depth, model_name, 'Guided_BackProp')

                        Guided_heatmap = visualize_rgb(guided_gradients_mask)
                        save_fig(Guided_heatmap, count, depth, model_name, 'Guided_BackProp')


                        # Guided Grad-CAM
                        guided_grad_cam = get_guided_grad_cam(grad_cam_mask, guided_gradients_mask)
                        save_gray_fig(guided_grad_cam, count, depth, model_name, 'GGrad_CAM')

                        GGrad_heatmap = visualize_rgb(guided_grad_cam)
                        save_fig(GGrad_heatmap, count, depth, model_name, 'GGrad_CAM')

                        count += 1
                        if count >= 5:
                                break             
        print("Done!")





