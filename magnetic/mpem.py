# import mag_manip as magmanip # forward and backward model
import mag_manip.mag_manip as magmanip # forward and backward model
import numpy as np
import torch
import os

class MPEMHandler:
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
        self.number_of_currents = number_of_currents
        self.device = device


    def get_currents_from_field_grad5(self, position, field, gradient):
        ''''
        Calculate the currents from the position, field and gradient5
        Args:
            position (torch.tensor): position of the sensor
            field (torch.tensor): field measured by the sensor
            gradient (torch.tensor): gradient5 of the magnetic field 
        Output:
            currents (torch.tensor): predicted currents
        '''
        position, field, gradient = self.check_convert_inputs_3(position), self.check_convert_inputs_3(field), self.check_convert_inputs_5(gradient) # Convert from torch to numpy
        currents = self.backward_model.computeCurrentsFromFieldGradient5(position, field, gradient)
        return torch.tensor(currents.reshape(self.number_of_currents), dtype=torch.float32, device=self.device)
    
    def get_currents_from_field_grad3(self, position, field, dipole, gradient):
        ''''
        Calculate the currents from the position, field, dipole and gradient3
        Args:
            position (torch.tensor): position of the sensor
            field (torch.tensor): field measured by the sensor
            dipole (torch.tensor): dipole of the magnet
            gradient (torch.tensor): gradient3 of the magnetic field (force on dipole)
        Output:
            currents (torch.tensor): predicted currents
        '''
        position, field, dipole, gradient = self.check_convert_inputs_3(position), self.check_convert_inputs_3(field), self.check_convert_inputs_3(dipole), self.check_convert_inputs_3(gradient) # Convert from torch to numpy
        currents = self.backward_model.computeCurrentsFromFieldDipoleGradient3(position, field, dipole, gradient)
        return torch.tensor(currents.reshape(self.number_of_currents), dtype=torch.float32, device=self.device)

    def get_field_gradient5(self, position, currents):
        '''
        Calculate the field gradient from the position and currents
        Args:
            position (np.array): position of the sensor
            currents (np.array): currents calculated from the sensor
        Output:
            field (torch.tensor): field prediction
            gradient (torch.tensor): gradient5 prediction
        '''
        position, currents = self.check_convert_inputs_3(position), self.check_convert_currents(currents) # Convert from torch to numpy
        field_gradient = self.forward_model.computeFieldGradient5FromCurrents(position, currents)
        field, gradient = torch.tensor(field_gradient[0:3], dtype=torch.float32, device=self.device), torch.tensor(field_gradient[3:], dtype=torch.float32, device=self.device)
        return field.reshape(3), gradient.reshape(5)
    

    
    def check_convert_inputs_3(self, input):
        '''
        Check if the inputs are valid & convert to numpy
        Args:
            input (torch.tensor): input to be checked
        Output:
            input (np.array): input converted to numpy
        '''
        if not isinstance(input, torch.Tensor):
            raise ValueError('Input should be a pytorch')
        input = input.cpu().numpy().astype(np.float64).reshape(3, 1)
        return input
    
    def check_convert_inputs_5(self, input):
        '''
        Check if the inputs are valid & convert to numpy
        Args:
            input (torch.tensor): input to be checked
        Output:
            input (np.array): input converted to numpy
        '''
        if not isinstance(input, torch.Tensor):
            raise ValueError('Input should be a pytorch tensor')
        input = input.cpu().numpy().astype(np.float64).reshape(5, 1)
        return input

    def check_convert_currents(self, input):
        '''
        Check if the inputs are valid & convert to numpy
        Args:
            input (torch.tensor): input to be checked
        Output:
            input (np.array): input converted to numpy
        '''
        if not isinstance(input, torch.Tensor):
            raise ValueError('Input should be a pytorch tensor')
        input = input.cpu().numpy().astype(np.float64).reshape(self.number_of_currents, 1)
        return input
