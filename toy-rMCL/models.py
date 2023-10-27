import torch
import torch.nn as nn
 
class smcl(nn.Module):

    def __init__(self,num_hypothesis,num_hidden_units=256):
        """Constructor for the multi-hypothesis network.

        Args:
            num_hypothesis (int): Number of output hypotheses.
            num_hidden_units (int, optional): _description_. Defaults to 256.
        """
        super(smcl, self).__init__()
        self.name = 'smcl'
        self.num_hypothesis = num_hypothesis
        self.fc1 = nn.Linear(1, num_hidden_units)  # input layer (1 inputs -> num_hidden_units hidden units)
        self.relu1 = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(num_hidden_units, num_hidden_units)  # hidden layer (num_hidden_units hidden units -> num_hidden_units hidden units)
        self.relu2 = nn.ReLU()  # ReLU activation function
        self.final_layers = nn.ModuleDict()
        
        for k in range(self.num_hypothesis) :  
            self.final_layers['hyp_'+'{}'.format(k)] = nn.Linear(in_features=num_hidden_units, out_features=2)

    def forward(self, x):
        """Forward pass of the multi-hypothesis network.

        Returns:
            hyp_stacked (torch.Tensor): Stacked hypotheses. Shape [batchxself.num_hypothesisxoutput_dim]
            confs (torch.Tensor): Confidence of each hypothesis (uniform for the classical sMCL model). Shape [batchxself.num_hypothesisx1]
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        outputs_hyps = []
        
        for k in range(self.num_hypothesis) :
             outputs_hyps.append((self.final_layers['hyp_'+'{}'.format(k)](x))) # Size [batchxoutput_dim]
            
        hyp_stacked = torch.stack(outputs_hyps, dim=-2) #Shape [batchxself.num_hypothesisxoutput_dim]
        assert hyp_stacked.shape == (x.shape[0], self.num_hypothesis, 2)
        confs = torch.ones_like(hyp_stacked[:,:,0]).unsqueeze(-1) #Shape [batchxself.num_hypothesisx1]
        return hyp_stacked, confs
    
class rmcl(nn.Module):
    
    def __init__(self,num_hypothesis):
        """Constructor for the multi-hypothesis network with confidence (rMCL).

        Args:
            num_hypothesis (int): Number of output hypotheses.
        """
        super(rmcl, self).__init__()
        self.name = 'rmcl'
        self.num_hypothesis = num_hypothesis
        num_hidden_units = 256
        self.fc1 = nn.Linear(1, num_hidden_units)  # input layer (1 inputs -> num_hidden_units hidden units)
        self.relu1 = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(num_hidden_units, num_hidden_units)  # hidden layer (num_hidden_units hidden units -> num_hidden_units hidden units)
        self.relu2 = nn.ReLU()  # ReLU activation function
        self.final_hyp_layers = nn.ModuleDict()
        self.final_conf_layers = nn.ModuleDict()
        
        for k in range(self.num_hypothesis) :  
            self.final_hyp_layers['hyp_'+'{}'.format(k)] = nn.Linear(in_features=num_hidden_units, out_features=2)
            self.final_conf_layers['hyp_'+'{}'.format(k)] = nn.Linear(in_features=num_hidden_units, out_features=1)

    def forward(self, x):
        """For pass of the multi-hypothesis network with confidence (rMCL).

        Returns:
            hyp_stacked (torch.Tensor): Stacked hypotheses. Shape [batchxself.num_hypothesisxoutput_dim]
            confs (torch.Tensor): Confidence of each hypothesis. Shape [batchxself.num_hypothesisx1]
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        outputs_hyps = []
        confidences = []
        
        for k in range(self.num_hypothesis) :
            outputs_hyps.append((self.final_hyp_layers['hyp_'+'{}'.format(k)](x))) # Size [batchxoutput_dim]
            confidences.append(torch.nn.Sigmoid()( self.final_conf_layers['hyp_'+'{}'.format(k)](x)))# Size [batchx1])

        hyp_stacked = torch.stack(outputs_hyps, dim=-2) #Shape [batchxself.num_hypothesisxoutput_dim]
        assert hyp_stacked.shape == (x.shape[0], self.num_hypothesis, 2)
        conf_stacked = torch.stack(confidences, dim=-2) #[batchxself.num_hypothesisx1]
        assert  conf_stacked.shape == (x.shape[0], self.num_hypothesis, 1)

        return hyp_stacked, conf_stacked