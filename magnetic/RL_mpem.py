# import mag_manip as magmanip # forward and backward model
import mag_manip.mag_manip as magmanip # forward and backward model
import numpy as np
import torch
import os

class RLMPEMHandler:
    '''
    Class to handle the currents and field calculations using the MPEM model.
    '''
    def __init__(self, calibration_path='Navion_1_2_Calibration.yaml', number_of_currents=3, device='cuda:0'):
        '''
        Initialize the MPEM model
        Args:
            calibration_path (str): path to the calibration file in the Calibration_data folder
        '''
        directory = os.path.dirname(os.path.abspath(__file__))
        calib_data_folder = os.path.join(directory, 'calibration')
        self.calibration_path = os.path.join(calib_data_folder, calibration_path)

        # Loading Forward Model
        self.forward_model = magmanip.ForwardModelMPEM()
        self.forward_model.setCalibrationFile(self.calibration_path)
        # Loading Backward Model
        self.backward_model = magmanip.BackwardModelMPEML2()
        self.backward_model.setCalibrationFile(self.calibration_path)
        self.device = device

    def get_currents_from_field(self, field: torch.Tensor) -> torch.Tensor:
        '''
        Get the currents from the field
        Args:
            field (torch.Tensor): magnetic field [num_envs, 3]
        Returns:
            currents (torch.Tensor): currents [num_envs, 3]
        '''
        # field = field.cpu().numpy().astype(np.float64)
        # field= field.T
        # field *= -1.0 # The field needs to be inverted because of the way the calibration file coordinate frame are defined
        # positions = np.zeros_like(field, dtype=np.float64)  # Assuming the field is at the origin
        # currents = self.backward_model.computeCurrentsFromFields(positions=positions, fields=field)
        # currents = currents.T
        # currents =  torch.tensor(currents, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            field_np = field.detach().cpu().numpy().astype(np.float64)
            field_np = field_np.T * -1.0
            positions = np.zeros_like(field_np)
            currents_np = self.backward_model.computeCurrentsFromFields(positions, field_np)
            currents_tensor = torch.from_numpy(currents_np.T).to(self.device, non_blocking=True).float()
            torch.cuda.empty_cache()
        return currents_tensor
    
    def get_field_from_currents(self, positions: torch.Tensor, currents: torch.Tensor) -> torch.Tensor:
        '''
        Get the field from the currents
        Args:
            positions (torch.Tensor): positions of the magnets [number_of_bodies, 3]
            currents (torch.Tensor): currents [1, 3]
        Returns:
            field (torch.Tensor): magnetic field [num_envs, 3]
        '''
        positions = positions.detach().cpu().numpy().astype(np.float64)
        currents = currents.detach().cpu().numpy().astype(np.float64)
        positions = positions.T
        currents = currents.T
        field = self.forward_model.computeFieldsFromCurrents(positions=positions, currents=currents)
        field = field.T
        field *= -1.0  # The field needs to be inverted because of the way the calibration file coordinate frame are defined
        field_tensor = torch.from_numpy(field).to(self.device, non_blocking=True).float()
        return field_tensor




