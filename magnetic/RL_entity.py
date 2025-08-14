import torch
from isaaclab.assets import Articulation
from .RL_mpem import RLMPEMHandler
from .force_torque_calc import MagneticForceTorque
from isaaclab.utils.math import quat_inv, quat_mul, quat_apply, convert_quat
from scipy.spatial.transform import Rotation as R
import torch
from typing import Union

@torch.jit.script
def covert_quat2dipolemoment(
    initial_quad: torch.Tensor,  # Initial quaternion [w,x,y,z]
    current_quad: torch.Tensor,  # Current quaternion [w,x,y,z]
    initial_magnetic_dipole: torch.Tensor  # Initial magnetic dipole [x,y,z]
) -> torch.Tensor:
    '''
    Convert the quaternion to a magnetic dipole moment
    Args:
        initial_quad (torch.tensor): initial quaternion [w,x,y,z]
        current_quad (torch.tensor): current quaternion [w,x,y,z]
        initial_magnetic_dipole (torch.tensor): initial magnetic dipole [x,y,z]
    Output:
        current_dipole_moment (torch.tensor): current magnetic dipole moment [x,y,z]
    '''
    current_rotation = quat_mul(quat_inv(initial_quad), current_quad).to(torch.float32)
    return quat_apply(quat=current_rotation, vec=initial_magnetic_dipole)

class RLMagneticEntity:
    '''
    Class to handle the magnetic entity
    '''
    def __init__(self, volume, remanence, direction, origins: torch.Tensor, magnet: Articulation, body_ids: list[int] = [0]):
        '''
        Initialize the magnetic entity
        Args:
            volume : volume of the magnet
            remanence : remanence of the magnet
            direction : direction of the magnet
            magnet : magnet object
        '''
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        volume, remanence = torch.tensor(volume, device=self.device, dtype=torch.float32), torch.tensor(remanence, device=self.device, dtype=torch.float32)
        self.body_ids = body_ids
        self.len_body_ids = len(self.body_ids)
        self.magnet = magnet
        self.mangetic_dipole_magnitude = self.calculate_magnetic_dipole_magnitude(volume, remanence)
        direction = (direction/torch.linalg.norm(direction))
        self.mpem_handler = RLMPEMHandler(calibration_path='Navion_1_2_Calibration.yaml',number_of_currents=3, device=self.device)
        self.magnetic_force_torque = MagneticForceTorque(device=self.device)
        self.initial_quad = self.get_current_quad().clone().detach()
        self.orginis = origins.detach().clone().unsqueeze(1).expand(-1, self.len_body_ids, -1)  #[num_envs, num_body_ids, 3]
        self.initial_magnetic_dipole = direction*self.mangetic_dipole_magnitude
        self.initial_magnetic_dipole = self.initial_magnetic_dipole.expand(self.orginis.shape[0]*self.len_body_ids, -1)  #[num_envs*len(body_ids), 3]

    def calculate_magnetic_dipole_magnitude(self, volume, remanence):
        '''
        Calculate the magnetic dipole magnitude
        Args:
            volume (torch.tensor): volume of the magnet
            remanence (float): remanence of the magnet
        Output:
            magnetic_dipole_magnitude (torch.tensor): magnetic dipole magnitude
        '''
        mu_0 = torch.tensor((4*torch.pi)*1e-7, device=self.device)
        return (1/mu_0)*volume*remanence
    
    def get_current_quad(self):
        '''
        Get the current quaternion [(num_body_ids*num_envs, (w,x,y,z)] of the magnet (in world frame)
        '''
        dipole_quat = self.magnet.data.body_link_state_w
        dipole_quat = dipole_quat[:, self.body_ids, 3:7].reshape((-1, 4)).to(torch.float32)
        return dipole_quat
        # return self.magnet.data.body_link_quat_w[:, self.body_ids, :].reshape((-1, 4)).to(torch.float32)
    
    def get_current_dipole_moment(self):
        '''
        Get the current dipole moment of the magnet
        '''
        current_quad = self.get_current_quad()
        return covert_quat2dipolemoment(self.initial_quad, current_quad, self.initial_magnetic_dipole)
    
    def get_position_and_dipole(self) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Get the position and dipole moment of the magnet
        Output:
            position (torch.tensor): position of the magnet [num_envs*len(body_ids), 3]
            dipole (torch.tensor): dipole moment of the magnet [num_envs*len(body_ids), 3]
        '''
        output = self.magnet.data.body_link_state_w
        position = output[:, self.body_ids, :3] - self.orginis
        position = position.reshape((-1, 3))  # Reshape to [num_env*len(body_ids), 3]
        quat = output[:, self.body_ids, 3:7].reshape((-1, 4)).to(torch.float32)
        dipole = covert_quat2dipolemoment(self.initial_quad, quat, self.initial_magnetic_dipole)
        return position, dipole

    def get_force_torque_from_currents(self, currents) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Calculate the force and torque from the currents
        Args:
            currents (torch.tensor): currents of the magnet
        '''
        positions, dipole = self.get_position_and_dipole()
        fields = self.get_field_from_currents(positions=positions, currents=currents)  # Get the fields from the currents
        torque = torch.cross(dipole, fields, dim=-1)  # Calculate torque as cross product of field and dipole
        torque = torque.reshape((-1, self.len_body_ids, 3))
        force = torch.zeros_like(torque)
        return force, torque

    
    def get_currents_from_field(self, field: torch.Tensor) -> torch.Tensor:
        '''
        Get the currents from the field
        Args:
            field (torch.Tensor): magnetic field [num_envs, 3]
        Returns:
            currents (torch.Tensor): currents [num_envs, 3]
        '''
        return self.mpem_handler.get_currents_from_field(field)

    
    def get_field_from_currents(self, positions: torch.Tensor, currents: torch.Tensor) -> torch.Tensor:
        '''
        Get the field from the currents
        Args:
            positions (torch.Tensor): positions of the magnets [num_envs * number_of_bodies, 3]
            currents (torch.Tensor): currents [num_envs, 3]
        Returns:
            field (torch.Tensor): magnetic field [num_envs, 3]
        '''
        # fields = []
        # for i in range(self.orginis.shape[0]):
        #     field = self.mpem_handler.get_field_from_currents(positions=positions[i*self.len_body_ids:(i+1)*self.len_body_ids, :], currents=currents[i, :])
        #     fields.append(field)
        # fields_torch = torch.stack(fields, dim=0).reshape((-1, 3))  # Reshape to [num_envs*len(body_ids), 3]
        # fields.clear()
        with torch.no_grad():
            fields = torch.zeros((positions.shape[0], 3), device=self.device)
            for i in range(self.orginis.shape[0]):
                idx_start, idx_end = i*self.len_body_ids, (i+1)*self.len_body_ids
                fields[idx_start:idx_end] = self.mpem_handler.get_field_from_currents(
                    positions=positions[idx_start:idx_end],
                    currents=currents[i]
                )
            torch.cuda.empty_cache()
        return fields
