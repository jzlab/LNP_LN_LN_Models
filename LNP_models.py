import torch
import torch.nn as nn
import numpy as np
import yaml

class RGCModel(nn.Module):
    """
    Generic Linear-Nonlinear (LN) and LN-LN model for RGC types.
    Adapts to dynamic input sizes based on checker_um and stimulus dimensions.
    """
    def __init__(self, cell_type, params_path='params.yaml'):
        super(RGCModel, self).__init__()
        self.cell_type = cell_type
        
        # Load parameters
        with open(params_path, 'r') as f:
            all_params = yaml.safe_load(f)
        
        self.grid = all_params['grid']
        self.params = all_params['cell_types'][cell_type]
        
        # Computation of dynamic parameters as requested
        self.checker_um = self.grid['checker_um']
        self.height = self.grid['height']
        self.width = self.grid['width']
        self.center_size = self.params['rf_um'] / self.checker_um
        self.surround_size = (2 * self.params['rf_um']) / self.checker_um
        
        # Model configuration (shared defaults based on RGC types)
        self.config = {
            'center_strength': 1.0,
            'surround_strength': -0.5,
            'alpha': 1.0,
            'gamma': 0.0,
            'frame_rate': 120,
            'memory_ms': 250,
            **self._get_type_specific_defaults()
        }
        
        self.dt = 1000.0 / self.config['frame_rate']
        self.memory_frames = int(self.config['memory_ms'] * self.config['frame_rate'] / 1000)

        # Precompute and register filters
        self.register_buffer('spatial_filter', self._create_spatial_filter())
        self.register_buffer('temporal_filter', self._create_temporal_filter())
        
        # Build 3D spatiotemporal filter and normalize
        st_filter = self.temporal_filter.view(-1, 1, 1) * self.spatial_filter.view(1, self.height, self.width)
        st_filter = st_filter / torch.norm(st_filter)
        self.register_buffer('st_filter', st_filter)

    def _get_type_specific_defaults(self):
        """Returns hardcoded defaults that are common for Parasol/Midget types."""
        is_parasol = 'Parasol' in self.cell_type
        is_on = 'ON' in self.cell_type

        # Nonlinearity beta
        beta = 2.0 if is_parasol else 1.0
        
        # Color mapping (moved from YAML to code)
        colors = {
            'ON_Parasol': 'red',
            'OFF_Parasol': 'orange',
            'ON_Midget': 'green',
            'OFF_Midget': 'blue'
        }
        color = colors.get(self.cell_type, 'black')

        # Subunit parameters (LN-LN cascade defaults from original code)
        if is_parasol:
            alpha_sub = 3.0 if is_on else 1.0
            beta_sub = 10.0
            gamma_sub = -0.3 if is_on else 0.0
            gamma_lnln = 5.0 if is_on else 2.0
        else: # Midget
            alpha_sub = 10.0 if is_on else 1.0
            beta_sub = 20.0 if is_on else 10.0
            gamma_sub = 1.0 if is_on else 0.0
            gamma_lnln = 5.0 if is_on else 2.0
            
        return {
            'beta': beta,
            'color': color,
            'alpha_sub': alpha_sub,
            'beta_sub': beta_sub,
            'gamma_sub': gamma_sub,
            'gamma_lnln': gamma_lnln
        }

    def _create_2d_gaussian(self, sigma):
        y = torch.arange(self.height).float()
        x = torch.arange(self.width).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        center_y, center_x = self.height / 2, self.width / 2
        gaussian = torch.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))
        return gaussian / torch.sum(gaussian)

    def _create_spatial_filter(self):
        center = self._create_2d_gaussian(self.center_size)
        surround = self._create_2d_gaussian(self.surround_size)
        return self.config['center_strength'] * center + self.config['surround_strength'] * surround

    def _create_temporal_filter(self):
        t_ms = torch.arange(self.memory_frames).float() * self.dt
        
        lobe1 = self.params['amp1'] * torch.exp(-((t_ms - self.params['peak1_ms'])**2) / (2 * self.params['width1_ms']**2))
        lobe2 = self.params['amp2'] * torch.exp(-((t_ms - self.params['peak2_ms'])**2) / (2 * self.params['width2_ms']**2))
        
        # Smooth onset
        onset_tau_ms = 15.0
        onset_window = 1.0 - torch.exp(-t_ms / onset_tau_ms)
        lobe1 *= onset_window
        lobe2 *= onset_window
        
        if self.cell_type.startswith('OFF'):
            return -lobe1 + lobe2
        return lobe1 - lobe2

    def apply_nonlinearity(self, x, alpha, beta, gamma, nonlinearity_type='soft_rectifier'):
        if nonlinearity_type == 'soft_rectifier':
            return alpha * torch.log(1.0 + torch.exp(beta * (x - gamma)))
        elif nonlinearity_type == 'relu':
            return alpha * torch.relu(beta * (x - gamma))
        return torch.clamp(x, min=0.0)

    def forward_lnp(self, stim_windowed, dt=8.3):
        N = stim_windowed.shape[0]
        s_flat = stim_windowed.reshape(N, -1).float()
        w_flat = self.st_filter.reshape(-1)
        generator_signal = torch.matmul(s_flat, w_flat)
        
        firing_rate = self.apply_nonlinearity(
            generator_signal, self.config['alpha'], self.config['beta'], self.config['gamma']
        )
        firing_rate = torch.clamp(firing_rate, max=700.0)
        
        return {
            'generator_signal': generator_signal,
            'firing_rate': firing_rate,
            'spikes': self.generate_spikes(firing_rate, dt),
            'spike_times': self.generate_spike_times(firing_rate, dt)
        }

    def forward_lnln(self, stim_windowed, dt=8.3, subunit_size=1, nonlinearity_type='soft_rectifier'):
        N, L, H, W = stim_windowed.shape
        u = torch.einsum('nlhw,l->nhw', stim_windowed.float(), self.temporal_filter)
        
        if subunit_size > 1:
            pool = nn.AvgPool2d(subunit_size, stride=1, padding=subunit_size//2)
            u_pooled = pool(u.unsqueeze(1)).squeeze(1) * (subunit_size**2)
        else:
            u_pooled = u
            
        h = self.apply_nonlinearity(
            u_pooled, self.config['alpha_sub'], self.config['beta_sub'], 
            self.config['gamma_sub'], nonlinearity_type
        )
        
        generator_signal = torch.einsum('nhw,hw->n', h, self.spatial_filter)
        
        firing_rate = self.apply_nonlinearity(
            generator_signal, self.config['alpha'], self.config['beta'],
            self.config.get('gamma_lnln', self.config['gamma']), nonlinearity_type
        )
        firing_rate = torch.clamp(firing_rate, max=700.0)
        
        return {
            'generator_signal': generator_signal, 'firing_rate': firing_rate,
            'spikes': self.generate_spikes(firing_rate, dt),
            'spike_times': self.generate_spike_times(firing_rate, dt),
            'u': u, 'h': h
        }

    def generate_spikes(self, firing_rate, dt=8.3):
        lambda_param = firing_rate * (dt / 1000.0)
        return torch.poisson(lambda_param).int()

    def generate_spike_times(self, firing_rate, dt=8.3):
        rates = firing_rate.detach().cpu().numpy()
        lam_per_bin = rates * (dt / 1000.0)
        counts = np.random.poisson(lam_per_bin)
        
        spike_times = []
        for i, k in enumerate(counts):
            if k > 0:
                bin_start, bin_end = i * dt, (i + 1) * dt
                spike_times.extend(np.random.uniform(bin_start, bin_end, int(k)))
        
        return np.sort(np.array(spike_times))

def load_all_models(params_path='params.yaml'):
    with open(params_path, 'r') as f:
        all_params = yaml.safe_load(f)
    return {ctype: RGCModel(ctype, params_path) for ctype in all_params['cell_types']}
