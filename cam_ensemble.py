# cam_ensemble.py
import torch
import torch.nn.functional as F

class CAMEnsemble:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self.gradients = {}
        
        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations['value'] = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0].detach()
            
        for layer in self.target_layers:
            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(backward_hook)

    def _grad_cam(self, input_tensor, target_category=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_category is None:
            target_category = torch.argmax(output, dim=1)
        
        # Batch-safe one-hot encoding
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_category.unsqueeze(1), 1.0)
        
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients['value']
        activations = self.activations['value']
        
        weights = F.adaptive_avg_pool2d(gradients, 1)
        cam = torch.sum(activations * weights, dim=1, keepdim=True)
        return F.relu(cam)

    def _grad_cam_pp(self, input_tensor, target_category=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_category is None:
            target_category = torch.argmax(output, dim=1)
        
        # Batch-safe one-hot encoding
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_category.unsqueeze(1), 1.0)
        
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients['value']
        activations = self.activations['value']
        
        # Grad-CAM++ calculations
        grads_power_2 = gradients.pow(2)
        grads_power_3 = gradients.pow(3)
        alpha = grads_power_2 / (2 * grads_power_2 + 
                torch.sum(activations * grads_power_3, dim=[2,3], keepdim=True) + 1e-6)
        
        weights = torch.sum(alpha * F.relu(gradients), dim=[2,3], keepdim=True)
        cam = torch.sum(activations * weights, dim=1, keepdim=True)
        return F.relu(cam)

    def _score_cam(self, input_tensor, target_category=None):
        activations = self.activations['value'].detach()
        b, c, h, w = activations.size()
        
        if target_category is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_category = torch.argmax(output, dim=1)

        scores = []
        for i in range(min(c, 32)):  # Limit channels for efficiency
            act = activations[:, i:i+1, :, :]
            act = (act - act.min()) / (act.max() - act.min() + 1e-6)
            
            upsampled_act = F.interpolate(act, input_tensor.shape[2:], mode='bilinear')
            masked_input = input_tensor * upsampled_act
            
            with torch.no_grad():
                output = self.model(masked_input)
                score = output.gather(1, target_category.unsqueeze(1))
            
            scores.append(score.view(-1, 1, 1, 1))
        
        scores = torch.cat(scores, dim=1)
        weights = F.softmax(scores, dim=1)
        cam = torch.sum(activations[:, :min(c, 32)] * weights, dim=1, keepdim=True)
        return F.relu(cam)

    def generate_ensemble_cam(self, input_tensor, target_category=None):
        grad_cam = self._grad_cam(input_tensor, target_category)
        grad_cam_pp = self._grad_cam_pp(input_tensor, target_category)
        score_cam = self._score_cam(input_tensor, target_category)
        
        combined_cam = (grad_cam + grad_cam_pp + score_cam) / 3
        combined_cam = combined_cam - combined_cam.min()
        combined_cam = combined_cam / (combined_cam.max() + 1e-6)
        
        if combined_cam.shape[2:] != input_tensor.shape[2:]:
            combined_cam = F.interpolate(combined_cam, input_tensor.shape[2:], mode='bilinear')
        
        return combined_cam
