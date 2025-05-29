import torch
from isaaclab.assets import RigidObject
from omni.physics.tensors.impl.api import RigidBodyView
from .mpem import MPEMHandler
from .force_torque_calc import MagneticForceTorque
from isaaclab.utils.math import quat_inv, quat_mul, quat_apply
from scipy.spatial.transform import Rotation as R
import torch
from typing import Union

class MagneticEntity:
    '''
    Class to handle the magnetic entity
    '''
    def __init__(self, volume, remanence, direction, magnet:Union[RigidObject, RigidBodyView]):
        '''
        Initialize the magnetic entity
        Args:
            volume : volume of the magnet
            remanence : remanence of the magnet
            direction : direction of the magnet
            magnet : magnet object
        '''
        self.is_rigidbody_view = isinstance(magnet, RigidBodyView)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        volume, remanence = torch.tensor(volume, device=self.device, dtype=torch.float32), torch.tensor(remanence, device=self.device, dtype=torch.float32)
        self.magnet = magnet
        direction = self.check_direction(direction)
        self.mu_0 = torch.tensor((4*torch.pi)*1e-7, device=self.device)
        self.mangetic_dipole_magnitude = self.calculate_magnetic_dipole_magnitude(volume, remanence)
        direction = direction/torch.linalg.norm(direction)
        self.initial_magnetic_dipole = direction*self.mangetic_dipole_magnitude
        self.mpem_handler = MPEMHandler(calibration_path='OctoMag_Calibration.yaml',number_of_currents=8, device=self.device)
        # self.mpem_handler = MPEMHandler(device=self.device)
        self.magnetic_force_torque = MagneticForceTorque(device=self.device)
        self.initial_quad = self.get_current_quad()

    def calculate_magnetic_dipole_magnitude(self, volume, remanence):
        '''
        Calculate the magnetic dipole magnitude
        Args:
            volume (torch.tensor): volume of the magnet
            remanence (float): remanence of the magnet
        Output:
            magnetic_dipole_magnitude (torch.tensor): magnetic dipole magnitude
        '''
        return (1/self.mu_0)*volume*remanence
    
    def get_quad(self):
        current_quad = self.get_current_quad()
        current_rotation = quat_mul(quat_inv(self.initial_quad), current_quad).to(torch.float32)
        return torch.tensor(current_rotation, device=self.device)
    
    def get_quad_scipy(self):
        current_quad = self.get_current_quad().cpu()
        current_rotation = R.from_quat(self.initial_quad.cpu()).inv() * R.from_quat(current_quad)
        return torch.tensor(current_rotation.as_quat(), device=self.device)
        
    def get_magnets_position(self):
        '''
        Get the position of the magnet
        '''
        # If using RigidBodyView, get the position from the view
        if self.is_rigidbody_view:
            transform = self.magnet.get_transforms()
            current_position = transform[:, :3]
            return current_position
        # If using RigidObject, get the position from the data
        return self.magnet.data.root_state_w[:, :3]
    
    def get_current_quad(self):
        '''
        Get the current quaternion [w,x,y,z] of the magnet (in world frame)
        '''
        # If using RigidBodyView, get the quaternion from the view
        if self.is_rigidbody_view:
            transform = self.magnet.get_transforms()
            current_quad = transform[:, 3:7]
            return current_quad
        # If using RigidObject, get the quaternion from the data
        return self.magnet.data.root_state_w[:, 3:7]
    
    def get_current_dipole_moment(self):
        '''
        Get the current dipole moment of the magnet
        '''
        current_quad = self.get_current_quad()
        current_rotation = quat_mul(quat_inv(self.initial_quad), current_quad).to(torch.float32)
        current_dipole_moment = quat_apply(quat=current_rotation, vec=self.initial_magnetic_dipole)
        # current_quad = self.get_current_quad().cpu()
        # current_rotation = R.from_quat(self.initial_quad.cpu()).inv() * R.from_quat(current_quad)
        # current_dipole_moment2 = torch.tensor(current_rotation.apply(self.initial_magnetic_dipole.cpu()).flatten(), device=self.device)
        return current_dipole_moment

    def get_force_torque_from_currents(self, currents):
        '''
        Calculate the force and torque from the currents
        Args:
            currents (torch.tensor): currents of the magnet
        '''
        if self.is_rigidbody_view:
            positions = self.get_magnets_position()
            forces, torques = torch.zeros(positions.shape[0], 3, device=self.device), torch.zeros(positions.shape[0], 3, device=self.device)
            for i, position in enumerate(positions):
                field, gradient = self.mpem_handler.get_field_gradient5(position, currents)
                current_dipole_moment = self.get_current_dipole_moment()[i]
                force = self.magnetic_force_torque.calculate_force(current_dipole_moment, gradient)
                torque = self.magnetic_force_torque.calculate_troque(current_dipole_moment, field)
                forces[i] = force
                torques[i] = torque
            return forces, torques
        else:
            position = self.get_magnets_position()
            field, gradient = self.mpem_handler.get_field_gradient5(position, currents)
            current_dipole_moment = self.get_current_dipole_moment()
            force = self.magnetic_force_torque.calculate_force(current_dipole_moment, gradient)
            torque = self.magnetic_force_torque.calculate_troque(current_dipole_moment, field)
            return force, torque
    
    def apply_force_torque_on_magnet(self, currents):
        '''
        Calculate and apply the force and torque on the magnet
        Args:
            currents (torch.tensor): currents of the magnet
        '''
        if self.is_rigidbody_view:
            forces, torques = self.get_force_torque_from_currents(currents)
            all_indices = torch.arange(self.magnet.count, device=self.device)
            self.magnet.apply_forces_and_torques_at_position(forces, torques, None, indices=all_indices, is_global=True)
        else:
            force, torque = self.get_force_torque_from_currents(currents)
            self.magnet.set_external_force_and_torque(force, torque)
            self.magnet.write_data_to_sim()
    
    def get_currents_from_field_gradient5(self, field, gradient):
        '''
        Get the currents from the field and gradient
        Args:
            field (torch.tensor): field of the magnet
            gradient (torch.tensor): gradient of the magnetic field
        Output:
            currents (torch.tensor): predicted currents
        '''
        position = self.get_magnets_position()
        if self.is_rigidbody_view:
            position = position[0]
        currents = self.mpem_handler.get_currents_from_field_grad5(position, field, gradient)
        return currents
        
    def get_currents_from_field_gradient3(self, field, gradient):
        '''
        Get the currents from the field and gradient
        Args:
            field (torch.tensor): field of the magnet
            gradient (torch.tensor): gradient of the magnetic field
        Output:
            currents (torch.tensor): predicted currents
        '''
        position = self.get_magnets_position()
        dipole = self.get_current_dipole_moment()
        if self.is_rigidbody_view:
            position = position[0]
            dipole = dipole[0]
        currents = self.mpem_handler.get_currents_from_field_grad3(position, field, dipole, gradient)
        return currents
    
    def get_field_gradient5(self, currents):
        '''
        Get the field and gradient from the currents
        Args:
            currents (torch.tensor): currents of the magnet
        Output:
            field (torch.tensor): field prediction
            gradient (torch.tensor): gradient5 prediction
        '''
        position = self.get_magnets_position()
        field, gradient = self.mpem_handler.get_field_gradient5(position, currents)
        return field, gradient
    
    def check_direction(self, direction):
        '''
        Check if the direction is valid
        Args:
            direction (torch.tensor): direction of the magnet
        '''
        if self.is_rigidbody_view:
            count = self.magnet.count
            if direction.shape != (count, 3) and direction.shape != (count, 3, 1):
                raise ValueError('Direction should have shape (RigidBodyView.count, 3) or (RigidBodyView.count, 3, 1)')
            if type(direction) != torch.Tensor:
                raise ValueError('Direction should be a torch tensor')
            return direction
        else:
            direction = direction.view(-1, 3)
            if direction.shape != (3,) and direction.shape != (3,1):
                raise ValueError('Direction should have shape (3,) or (3,1)')
            if type(direction) != torch.Tensor:
                raise ValueError('Direction should be a torch tensor')
            return direction
        

        



        