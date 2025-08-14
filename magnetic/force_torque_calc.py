import torch

class MagneticForceTorque():
    def __init__(self, device):
        self.device = device
        
    def calculate_force(self, magnetic_dipole, gradient):
        '''
        Calculate the force from the magnetic dipole and gradient
        Args:
            magnetic_dipole (troch.tensor): magnetic dipole of the magnet
            gradient (torch.tensor): gradient of the magnetic field
        '''
        self.check_inputs_3(magnetic_dipole), self.check_inputs_5(gradient)
        [mx, my, mz] = magnetic_dipole
        return  compute_force(mx, my, mz, gradient)
    
    def calculate_troque(self, magnetic_dipole, field):
        '''
        Calculate the torque from the magnetic dipole and field
        Args:
            magnetic_dipole (troch.tensor): magnetic dipole of the magnet
            field (torch.tensor): field of the magnet
        '''
        self.check_inputs_3(magnetic_dipole), self.check_inputs_3(field)
        return compute_torque(magnetic_dipole, field)
    
    def check_inputs_3(self, input):
        '''
        Check if the inputs are valid
        Args:
            input (torch.tensor): input to be checked
        '''
        if not isinstance(input, torch.Tensor):
            raise ValueError('Input should be a torch tensor')
        if input.shape != (3,) and input.shape != (3,1) and input.shape != (1,3):
            raise ValueError('Input should have shape (3,) or (3,1)')
    
    def check_inputs_5(self, input):
        '''
        Check if the inputs are valid
        Args:
            input (torch.tensor): input to be checked
        '''
        if not isinstance(input, torch.Tensor):
            raise ValueError('Input should be a torch tensor')
        if input.shape != (5,) and input.shape !=(5,1) and input.shape != (1,5):
            raise ValueError('Input should have shape (5,) or (5,1)')
    

@torch.jit.script
def compute_force(
    mx: torch.Tensor,
    my: torch.Tensor,
    mz: torch.Tensor,
    gradient: torch.Tensor,
) -> torch.Tensor:
    """Compute the force from the magnetic dipole and gradient."""
    row0 = torch.stack([mx, my, mz, torch.tensor(0., device=gradient.device), torch.tensor(0., device=gradient.device)])
    row1 = torch.stack([torch.tensor(0., device=gradient.device), mx, torch.tensor(0., device=gradient.device), my, mz])
    row2 = torch.stack([-mz, torch.tensor(0., device=gradient.device), mx, -mz, my])
    force_matrix = torch.stack([row0, row1, row2])
    force = torch.matmul(force_matrix, gradient)
    return force.reshape(1, 3)

def compute_torque(
    magnetic_dipole: torch.Tensor,
    field: torch.Tensor,
    ) -> torch.Tensor:
    """Compute the torque from the magnetic dipole and field."""
    return torch.linalg.cross(magnetic_dipole, field).reshape(1, 3)
