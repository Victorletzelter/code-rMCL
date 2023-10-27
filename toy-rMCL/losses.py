
import torch
from torch.nn.modules.loss import _Loss

class mhloss(_Loss):
    """Class for multi-hypothesis (i.e., Winner-Takes-Loss variants) losses.
    """

    def __init__(self,            
                 reduction='mean',
                 mode = 'wta',
                 top_n = 1,
                 distance = 'euclidean',
                 epsilon=0.05,
                 single_target_loss=False) -> None:
        """Constructor for the multi-hypothesis loss.

        Args:
            reduction (str, optional): Type of reduction performed. Defaults to 'mean'.
            mode (str, optional): Winner-takes-all variant ('wta', 'wta-relaxed', 'wta-top-n') to choose. Defaults to 'wta'.
            top_n (int, optional): Value of n when applying the top_n variant. Defaults to 1.
            distance (str, optional): Underlying distance to use for the WTA computation. Defaults to 'euclidean'.
            epsilon (float, optional):  Value of epsilon when applying the wta-relaxed variant. Defaults to 0.05.
            single_target_loss (bool, optional): Whether to perform single target update (used in ensemble_mode). Defaults to False.
        """
        super(mhloss, self).__init__(reduction)

        self.mode = mode
        self.top_n = top_n
        self.distance = distance
        self.epsilon = epsilon
        self.single_target_loss = single_target_loss
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) :
        """Forward pass for the multi-hypothesis loss. 

        Args:
            predictions (torch.Tensor): Tensor of shape [batchxself.num_hypothesisxoutput_dim]
            targets (torch.Tensor,torch.Tensor): Tuple of shape [batch,Max_targets],[batch,Max_targets,output_dim], where Max_targets is the maximum number of targets for each input.

        Returns:
            loss (torch.tensor)
        """
        hyps_pred_stacked,_ = predictions #Shape [batchxself.num_hypothesisxoutput_dim]
        target_position, source_activity_target = targets #Shape [batch,Max_targets,output_dim],[batch,Max_targets,1]

        losses = torch.tensor(0.)
                
        source_activity_target = source_activity_target[:,:].detach()
        target_position =target_position[:,:,:].detach()
        
        loss=self.wta_loss(hyps_pred_stacked=hyps_pred_stacked, 
                                                            source_activity_target=source_activity_target, 
                                                            target_position=target_position, 
                                                            mode=self.mode,
                                                            top_n=self.top_n, 
                                                            distance=self.distance,
                                                            epsilon=self.epsilon,
                                                            single_target_loss=self.single_target_loss)
        losses = torch.add(losses,loss)

        return losses
    
    def wta_loss(self, hyps_pred_stacked, source_activity_target, target_position, mode='wta',top_n=1, distance='euclidean', epsilon=0.05, single_target_loss = False):
        """Winner takes all loss computation and its variants.

        Args:
            hyps_pred_stacked (torch.tensor): Input tensor of shape (batch,num_hyps,output_dim)
            source_activity_target torch.tensor): Input tensor of shape (batch,Max_targets)
            target_position (torch.tensor): Input tensor of shape (batch,Max_targets,output_dim)
            mode (str, optional): Variant of the classical WTA chosen. Defaults to 'wta'.
            top_n (int, optional): Top_n winner in the Evolving WTA mode. Defaults to 1.
            distance (str, optional): Underlying distance to use. Defaults to 'euclidean'.

        Returns:
            loss (torch.tensor)
        """
        
        filling_value = 1000 #Large number (on purpose) ; computational trick to ignore the "inactive targets".  
        # whenever the sources are not active, as the source_activity is not to be deduced by the model is these settings. 
        num_hyps = hyps_pred_stacked.shape[1]
        batch = source_activity_target.shape[0]
        Max_targets = source_activity_target.shape[1]
        # output_dim = target_position.shape[2]
        
        #1st padding related to the inactive sources, not considered in the error calculation (with high error values)
        mask_inactive_sources = source_activity_target == 0
        mask_inactive_sources = mask_inactive_sources.expand_as(target_position)
        target_position[mask_inactive_sources] = filling_value #Shape [batch,Max_targets,output_dim]
        
        #We can check whether the operation is performed correctly
        # assert (source_activity_target.sum(axis=1).all()==(target_position[:,:,0]!=filling_value).sum(axis=1).all())
        # assert (source_activity_target.sum(axis=1).all()==(target_position[:,:,1]!=filling_value).sum(axis=1).all())
        
        #The ground truth tensor created is of shape [batch,Max_targets,num_hyps,output_dim], such that each of the 
        # tensors gts[batch,i,num_hypothesis,output_dim] contains duplicates of target_position along the num_hypothesis
        # dimension. Note that for some values of i, gts[batch,i,num_hypothesis,output_dim] may contain inactive sources, and therefore 
        # gts[batch,i,j,output_dim] will be filled with filling_value (defined above) for each j in the hypothesis dimension.
        gts =  target_position.unsqueeze(2).repeat(1,1,num_hyps,1) #Shape [batch,Max_targets,num_hypothesis,output_dim]
        
        # assert gts.shape==(batch,Max_targets,num_hyps,output_dim)
        
        #We duplicate the hyps_stacked with a new dimension of shape Max_targets
        hyps_pred_stacked_duplicated = hyps_pred_stacked.unsqueeze(1).repeat(1,Max_targets,1,1) #Shape [batch,Max_targets,num_hypothesis,output_dim]

        # assert hyps_pred_stacked_duplicated.shape==(batch,Max_targets,num_hyps,output_dim)

        if distance=='euclidean' :
            #### With euclidean distance
            diff = torch.square(hyps_pred_stacked_duplicated-gts) #Shape [batch,Max_targets,num_hypothesis,output_dim]
            channels_sum = torch.sum(diff, dim=3) #Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_targets,num_hypothesis]
            
            # assert channels_sum.shape == (batch,Max_targets,num_hyps)
            # assert (channels_sum>=0).all()
            
            eps = 0.001
            dist_matrix = torch.sqrt(channels_sum + eps)  #Distance matrix [batch,Max_targets,num_hypothesis]
            
            # assert dist_matrix.shape == (batch,Max_targets,num_hyps)
            
        sum_losses = torch.tensor(0.)

        if mode == 'wta': 

            if single_target_loss==True:
                wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_targets] 
                wta_dist_matrix, idx_source_selected = torch.min(wta_dist_matrix,dim=1) 
                wta_dist_matrix = wta_dist_matrix.unsqueeze(-1) #[batch,1]
                # assert wta_dist_matrix.shape == (batch,1)
                mask = wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values. 
                wta_dist_matrix = wta_dist_matrix*mask #[batch,1], we select only the active sources.
                count_non_zeros = torch.sum(mask!=0) #We count the number of actives sources for the computation of the mean (below).

            else : 
                wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_targets] 
                mask = wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values. 
                wta_dist_matrix = wta_dist_matrix*mask #[batch,Max_targets], we select only the active sources. 
                count_non_zeros = torch.sum(mask!=0) #We count the number of actives sources for the computation of the mean (below). 
            
            if count_non_zeros>0 : 
                loss = torch.sum(wta_dist_matrix)/count_non_zeros #We compute the mean of the diff. 
            else :
                loss = torch.tensor(0.)    
            
            sum_losses = torch.add(sum_losses, loss) 
            
        elif mode == 'wta-relaxed':
        
            #We compute the loss for the "best" hypothesis. 
            
            wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_targets], idx_selected of shape [batch,Max_targets].
            
            # assert wta_dist_matrix.shape == (batch,Max_targets)
            # assert idx_selected.shape == (batch,Max_targets)
            
            mask = wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with
            wta_dist_matrix = wta_dist_matrix*mask #Shape [batch,Max_targets] ; we select only the active sources. 
            count_non_zeros_1 = torch.sum(mask!=0) #We count the number of actives sources as a sum over the batch for the computation of the mean (below).

            if count_non_zeros_1>0 : 
                loss0 = torch.multiply(torch.sum(wta_dist_matrix)/count_non_zeros_1, 1 - epsilon) #Scalar (average with coefficient)
            else :
                loss0 = torch.tensor(0.)    

            #We then the find the other hypothesis, and compute the epsilon weighted loss for them
            
            # At first, we remove hypothesis corresponding to "fake" ground-truth.         
            large_mask = dist_matrix <= filling_value # We remove entries corresponding to "fake"/filled ground truth in the tensor dist_matrix on
            # which the min operator was not already applied. Shape [batch,Max_targets,num_hypothesis]
            dist_matrix = dist_matrix*large_mask # Shape [batch,Max_targets,num_hypothesis].
            
            # We then remove the hypothesis selected above (with minimum dist)
            mask_selected = torch.zeros_like(dist_matrix,dtype=bool) #Shape [batch,Max_targets,num_hypothesis]
            mask_selected.scatter_(2, idx_selected.unsqueeze(-1), 1) # idx_selected new shape: [batch,Max_targets,1]. 
            # The assignement mask_selected[i,j,idx_selected[i,j]]=1 is performed. 
            # Shape of mask_selected: [batch,Max_targets,num_hypothesis]
            
            assert mask_selected.shape == (batch,Max_targets,num_hyps)
            
            mask_selected = ~mask_selected #Shape [batch,Max_targets,num_hypothesis], we keep only the hypothesis which are not the minimum.
            dist_matrix = dist_matrix * mask_selected #Shape [batch,Max_targets,num_hypothesis]
            
            # Finally, we compute the loss
            count_non_zeros_2 = torch.sum(dist_matrix!=0)

            if count_non_zeros_2 > 0 :
                loss = torch.multiply(torch.sum(dist_matrix)/count_non_zeros_2, epsilon) #Scalar for each hyp
            else : 
                loss = torch.tensor(0.)
            
            sum_losses = torch.add(sum_losses, loss)
            sum_losses = torch.add(sum_losses, loss0)
            
        elif mode == 'wta-top-n' and top_n > 1:
            
            # dist_matrix.shape == (batch,Max_targets,num_hyps)
            # wta_dist_matrix of shape [batch,Max_targets]
            
            dist_matrix = torch.multiply(dist_matrix, -1) # Shape (batch,Max_targets,num_hyps) 
            top_k, indices = torch.topk(input=dist_matrix, k=top_n, dim=-1) #top_k of shape (batch,Max_targets,top_n), indices of shape (batch,Max_targets,top_n) 
            dist_matrix_min = torch.multiply(top_k,-1) 
            
            mask = dist_matrix_min <= filling_value/2 # We create a mask of shape [batch,Max_targets,top_n] for only selecting the actives sources, i.e. those which were not filled with fake values. 
            assert mask[:,:,0].all() == mask[:,:,-1].all() # This mask related should be constant in the third dimension.
            
            dist_matrix_min = dist_matrix_min*mask # [batch,Max_targets,top_n], we select only the active sources.  
            assert dist_matrix_min.shape == (batch,Max_targets,top_n)
            
            count_non_zeros = torch.sum(mask[:,:,0]!=0) # We count the number of entries (in the first two dimensions) for which the mask is different from zero. 
            
            for i in range(top_n):
                
                assert count_non_zeros == torch.sum(mask[:,:,i]!=0) # We count the number of entries for which the mask is different from zero. 
                
                if count_non_zeros > 0 :
                    loss  = torch.multiply(torch.sum(dist_matrix_min[:, :, i])/count_non_zeros, 1.0)
                else :
                    loss = torch.tensor(0.)
                
                sum_losses = torch.add(sum_losses, loss)
                
            sum_losses = sum_losses / top_n
            
        return sum_losses
    
class rmcl_loss(_Loss):
    """Class for rMCL loss (and variants).
    """

    __constants__ = ['reduction']

    def __init__(self,
                 reduction='mean',
                 mode = 'wta',
                 top_n = 1,
                 distance = 'euclidean',
                 epsilon=0.05,
                 conf_weight = 1,
                 rejection_method = 'uniform_negative',
                 number_unconfident = 1) -> None:
        """Constructor for the rMCL loss.

        Args:
            reduction (str, optional): Type of reduction performed. Defaults to 'mean'.
            mode (str, optional): Winner-takes-all variant ('wta', 'wta-relaxed', 'wta-top-n') to choose. Defaults to 'wta'.
            top_n (int, optional): Value of n when applying the top_n variant. Defaults to 1.
            distance (str, optional): Underlying distance to use for the WTA computation. Defaults to 'euclidean'.
            epsilon (float, optional): Value of epsilon when applying the wta-relaxed variant. Defaults to 0.05.
            conf_weight (int, optional): Weight of the confidence loss (beta parameter). Defaults to 1.
            rejection_method (str, optional): Type of rejection, i.e., update of the negative hypothesis to perform. Defaults to 'uniform_negative'.
            number_unconfident (int, optional): Number of negative hypothesis to update when the rejection method is 'uniform_negative'. Defaults to 1.
        """

        super(rmcl_loss, self).__init__(reduction)
        self.mode = mode
        self.top_n = top_n
        self.distance = distance
        self.epsilon = epsilon
        self.conf_weight = conf_weight
        self.rejection_method = rejection_method
        self.number_unconfident = number_unconfident
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) :
        """Forward pass for the Multi-hypothesis rMCL Loss. 

        Args:
            predictions (torch.Tensor): Tuple of tensors of shape [batchxself.num_hypothesisxoutput_dim],[batchxself.num_hypothesisx1]
            targets (torch.Tensor,torch.Tensor): Tuple of tensors of shape [batch,Max_targets],[batch,Max_targets,output_dim]

        Returns:
            loss (torch.tensor)
        """

        hyps_pred_stacked, conf_pred_stacked = predictions #Shape [batchxself.num_hypothesisxoutput_dim], [batchxself.num_hypothesisx1]
        target_position, source_activity_target = targets #Shape [batch,Max_targets,output_dim],[batch,Max_targets,1]

        losses = torch.tensor(0.)
                
        source_activity_target = source_activity_target[:,:].detach()
        target_position =target_position[:,:,:].detach()
        
        loss=self.rmcl_loss(hyps_pred_stacked=hyps_pred_stacked, 
                                                        conf_pred_stacked=conf_pred_stacked,
                                                        source_activity_target=source_activity_target, 
                                                        target_position=target_position, 
                                                        mode=self.mode,
                                                        top_n=self.top_n, 
                                                        distance=self.distance,
                                                        epsilon=self.epsilon)
        losses = torch.add(losses,loss)

        return losses
    
    def rmcl_loss(self, hyps_pred_stacked, conf_pred_stacked, source_activity_target, target_position, mode='wta',top_n=1, distance='euclidean', epsilon=0.05, conf_weight = 1.,rejection_method='uniform_negative',number_unconfident = 3):
        """Winner takes all loss computation and its variants.

        Args:
            hyps_pred_stacked (torch.tensor): Input tensor of shape (batch,num_hyps,output_dim)
            source_activity_target torch.tensor): Input tensor of shape (batch,Max_targets)
            conf_pred_stacked (torch.tensor): Input tensor of shape (batch,num_hyps,1)
            target_position (torch.tensor): Input tensor of shape (batch,Max_targets,output_dim)
            mode (str, optional): Variant of the classical WTA chosen. Defaults to 'epe'.
            top_n (int, optional): top_n winner in the Evolving WTA mode. Defaults to 1.
            distance (str, optional): _description_. Defaults to 'euclidean'.

        Returns:
            loss (torch.tensor)
        """
        
        filling_value = 1000 #Large number (on purpose) ; computational trick to ignore the "fake" ground truths.
        # whenever the sources are not active, as the source_activity is not to be deduced by the model is these settings. 
        num_hyps = hyps_pred_stacked.shape[1]
        batch = source_activity_target.shape[0]
        Max_targets = source_activity_target.shape[1]
        # output_dim = target_position.shape[2]

        # assert num_hyps > number_unconfident, "The number of hypothesis is too small comparing to the number of unconfident hypothesis selected in the scoring" # We check that the number of hypothesis is higher than the number of "negative" hypothesis sampled. 
        
        #1st padding related to the inactive sources, not considered in the error calculation (with high error values)
        mask_inactive_sources = source_activity_target == 0
        mask_inactive_sources = mask_inactive_sources.expand_as(target_position)
        target_position[mask_inactive_sources] = filling_value #Shape [batch,Max_targets,output_dim]
        
        #We can check whether the operation is performed correctly
        # assert (source_activity_target.sum(axis=1).all()==(target_position[:,:,0]!=filling_value).sum(axis=1).all())
        # assert (source_activity_target.sum(axis=1).all()==(target_position[:,:,1]!=filling_value).sum(axis=1).all())
        
        #The ground truth tensor created is of shape [batch,Max_targets,num_hyps,output_dim], such that each of the 
        # tensors gts[batch,i,num_hypothesis,output_dim] contains duplicates of target_position along the num_hypothesis
        # dimension. Note that for some values of i, gts[batch,i,num_hypothesis,output_dim] may contain inactive sources, and therefore 
        # gts[batch,i,j,output_dim] will be filled with filling_value (defined above) for each j in the hypothesis dimension.
        gts =  target_position.unsqueeze(2).repeat(1,1,num_hyps,1) #Shape [batch,Max_targets,num_hypothesis,output_dim]
        
        # assert gts.shape==(batch,Max_targets,num_hyps,output_dim)
        
        #We duplicate the hyps_stacked with a new dimension of shape Max_targets
        hyps_pred_stacked_duplicated = hyps_pred_stacked.unsqueeze(1).repeat(1,Max_targets,1,1) #Shape [batch,Max_targets,num_hypothesis,output_dim]

        # assert hyps_pred_stacked_duplicated.shape==(batch,Max_targets,num_hyps,output_dim)
        
        eps = 0.001

        ### Management of the confidence part
        conf_pred_stacked = torch.squeeze(conf_pred_stacked,dim=-1) #(batch,num_hyps), predicted confidence scores for each hypothesis.
        gt_conf_stacked_t = torch.zeros_like(conf_pred_stacked, device=conf_pred_stacked.device) #(batch,num_hyps), will contain the ground-truth of the confidence scores. 
        
        # assert gt_conf_stacked_t.shape == (batch,num_hyps)

        if distance=='euclidean' :
            #### With euclidean distance
            diff = torch.square(hyps_pred_stacked_duplicated-gts) #Shape [batch,Max_targets,num_hyps,output_dim]
            channels_sum = torch.sum(diff, dim=3) #Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_targets,num_hypothesis]
            
            dist_matrix = torch.sqrt(channels_sum + eps)  #Distance matrix [batch,Max_targets,num_hyps]
            
            # assert dist_matrix.shape == (batch,Max_targets,num_hyps)
            
        sum_losses = torch.tensor(0.)

        if mode == 'wta': 
            
            # We select the best hypothesis for each source
            wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_targets]

            mask = wta_dist_matrix <= filling_value/2 #We create a mask of shape [batch,Max_targets] for only selecting the actives sources, i.e. those which were not filled with fake values. 
            wta_dist_matrix = wta_dist_matrix*mask #[batch,Max_targets], we select only the active sources.

            # Create tensors to index batch and Max_targets dimensions
            batch_indices = torch.arange(batch, device=conf_pred_stacked.device)[:, None].expand(-1, Max_targets) # Shape (batch, Max_targets)

            # We set the confidences of the selected hypotheses.
            gt_conf_stacked_t[batch_indices[mask], idx_selected[mask]] = 1 #Shape (batch,num_hyps)

            count_non_zeros = torch.sum(mask!=0) #We count the number of actives sources for the computation of the mean (below). 
            
            if count_non_zeros>0 : 
                loss = torch.sum(wta_dist_matrix)/count_non_zeros #We compute the mean of the diff.
                
                selected_confidence_mask = gt_conf_stacked_t == 1 # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores which
                # will be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.   
                unselected_mask = ~selected_confidence_mask # (batch,num_hyps), mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.  

                if rejection_method=='uniform_negative' : 
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack([torch.multinomial(unselected_mask[b_idx].float(), number_unconfident, replacement=False) for b_idx in range(batch)]) #(batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(-1, number_unconfident) #(batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    gt_conf_stacked_t[batch_indices, unconfident_indices] = 0 #(Useless) Line added for the sake of completness. 

                elif rejection_method=='all' :

                    selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(conf_pred_stacked[selected_confidence_mask], gt_conf_stacked_t[selected_confidence_mask])
               
            else :
                loss = torch.tensor(0.) 
                confidence_loss = torch.tensor(0.)   
            
            sum_losses = torch.add(sum_losses, loss)  
            sum_losses = torch.add(sum_losses, conf_weight*confidence_loss)
            
        elif mode == 'wta-relaxed':
        
            #We compute the loss for the "best" hypothesis but also for the others with weight epsilon.  
            
            wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_targets], idx_selected of shape [batch,Max_targets].
            
            # assert wta_dist_matrix.shape == (batch,Max_targets)
            # assert idx_selected.shape == (batch,Max_targets)
            
            mask = wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with
            wta_dist_matrix = wta_dist_matrix*mask #Shape [batch,Max_targets] ; we select only the active sources. 
            count_non_zeros_1 = torch.sum(mask!=0) #We count the number of actives sources as a sum over the batch for the computation of the mean (below).

            ### Confidence management
            # Create tensors to index batch and Max_targets dimensions
            batch_indices = torch.arange(batch)[:, None].expand(-1, Max_targets) # Shape (batch, Max_targets)

            # We set the confidence of the selected hypothesis
            gt_conf_stacked_t[batch_indices[mask], idx_selected[mask]] = 1 #Shape (batch,num_hyps)
            ###

            if count_non_zeros_1>0 : 
                loss0 = torch.multiply(torch.sum(wta_dist_matrix)/count_non_zeros_1, 1 - epsilon) #Scalar (average with coefficient)
                
                selected_confidence_mask = gt_conf_stacked_t == 1 # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores which
                # will be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.   
                unselected_mask = ~selected_confidence_mask # (batch,num_hyps), mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.  

                if rejection_method=='uniform_negative' : 
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack([torch.multinomial(unselected_mask[b_idx].float(), number_unconfident, replacement=False) for b_idx in range(batch)]) #(batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(-1, number_unconfident) #(batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    gt_conf_stacked_t[batch_indices, unconfident_indices] = 0 #(Useless) Line added for the sake of completness. 

                elif rejection_method=='all' :

                    selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                assert conf_pred_stacked.all()>0, "The original tensor was affected by the modification" # To check that the original tensor was not affected by the modification. 
                
                ### Uncomment the following lines to check that the selected_confidence_mask is correct in term of number of selected hypothesis.
                if rejection_method =='uniform_negative' :
                    assert selected_confidence_mask.sum() == batch*number_unconfident+torch.sum(gt_conf_stacked_t==1), "The number of selected hypothesis is not correct."

                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(conf_pred_stacked[selected_confidence_mask], gt_conf_stacked_t[selected_confidence_mask])
               
            else :
                loss0 = torch.tensor(0.) 
                confidence_loss = torch.tensor(0.)

            #We then the find the other hypothesis, and compute the epsilon weighted loss for them
            
            # At first, we remove hypothesis corresponding to "fake" ground-truth.         
            large_mask = dist_matrix <= filling_value # We remove entries corresponding to "fake"/filled ground truth in the tensor dist_matrix on
            # which the min operator was not already applied. Shape [batch,Max_targets,num_hypothesis]
            dist_matrix = dist_matrix*large_mask # Shape [batch,Max_targets,num_hypothesis].
            
            # We then remove the hypothesis selected above (with minimum dist)
            mask_selected = torch.zeros_like(dist_matrix,dtype=bool) #Shape [batch,Max_targets,num_hypothesis]
            mask_selected.scatter_(2, idx_selected.unsqueeze(-1), 1) # idx_selected new shape: [batch,Max_targets,1]. 
            # The assignement mask_selected[i,j,idx_selected[i,j]]=1 is performed. 
            # Shape of mask_selected: [batch,Max_targets,num_hypothesis]
            
            # assert mask_selected.shape == (batch,Max_targets,num_hyps)
            
            mask_selected = ~mask_selected #Shape [batch,Max_targets,num_hypothesis], we keep only the hypothesis which are not the minimum.
            dist_matrix = dist_matrix * mask_selected #Shape [batch,Max_targets,num_hypothesis]
            
            # Finally, we compute the loss
            count_non_zeros_2 = torch.sum(dist_matrix!=0)

            if count_non_zeros_2 > 0 :
                epsilon_loss = torch.multiply(torch.sum(dist_matrix)/count_non_zeros_2, epsilon) #Scalar for each hyp
            else : 
                epsilon_loss = torch.tensor(0.)
            
            sum_losses = torch.add(sum_losses, epsilon_loss) # Loss for the unselected (i.e., not winners) hypothesis (epsilon weighted)
            sum_losses = torch.add(sum_losses, loss0) # Loss for the selected (i.e., the winners) hypothesis (1-epsilon weighted)
            sum_losses = torch.add(sum_losses, conf_weight*confidence_loss) # Loss for the confidence prediction. 
            
        elif mode == 'wta-top-n' and top_n > 1:
            
            # dist_matrix.shape == (batch,Max_targets,num_hyps)
            # wta_dist_matrix of shape [batch,Max_targets]
            
            dist_matrix = torch.multiply(dist_matrix, -1) # Shape (batch,Max_targets,num_hyps) 
            top_k, indices = torch.topk(input=dist_matrix, k=top_n, dim=-1) #top_k of shape (batch,Max_targets,top_n), indices of shape (batch,Max_targets,top_n) 
            dist_matrix_min = torch.multiply(top_k,-1) 
            
            mask = dist_matrix_min <= filling_value/2 # We create a mask of shape [batch,Max_targets,top_n] for only selecting the actives sources, i.e. those which were not filled with fake values. 
            # assert mask[:,:,0].all() == mask[:,:,-1].all() # This mask related should be constant in the third dimension.
            
            dist_matrix_min = dist_matrix_min*mask # [batch,Max_targets,top_n], we select only the active sources.  
            # assert dist_matrix_min.shape == (batch,Max_targets,top_n)
            
            count_non_zeros = torch.sum(mask[:,:,0]!=0) # We count the number of entries (in the first two dimensions) for which the mask is different from zero. 
            
            ### Confidence management
            # Create tensors to index batch and Max_targets and top-n dimensions. 
            batch_indices = torch.arange(batch)[:, None, None].expand(-1, Max_targets,top_n) # Shape (batch, Max_targets,top_n)
            # We set the confidence of the selected hypothesis
            gt_conf_stacked_t[batch_indices[mask], indices[mask]] = 1 #Shape (batch,num_hyps)
            selected_confidence_mask = gt_conf_stacked_t == 1 # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores 
            # to be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.
            ###
            
            for i in range(top_n):

                # assert count_non_zeros == torch.sum(mask[:,:,i]!=0) # We count the number of entries for which the mask is different from zero. 
            
                if count_non_zeros>0 : 
                    loss  = torch.multiply(torch.sum(dist_matrix_min[:, :, i])/count_non_zeros, 1.0)

                else :
                    loss = torch.tensor(0.) 

                sum_losses = torch.add(sum_losses, loss/top_n)

            if count_non_zeros>0 : 
              
                unselected_mask = ~selected_confidence_mask # (batch,num_hyps), mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.  

                if rejection_method=='uniform_negative' : 
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack([torch.multinomial(unselected_mask[b_idx].float(), number_unconfident, replacement=False) for b_idx in range(batch)]) #(batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(-1, number_unconfident) #(batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    gt_conf_stacked_t[batch_indices, unconfident_indices] = 0 #(Useless) Line added for the sake of completness. 

                elif rejection_method=='all' :

                    selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                # assert conf_pred_stacked.all()>0, "The original tensor was affected by the modification" # To check that the original tensor was not affected by the modification. 
                
                ### Uncomment the following lines to check that the selected_confidence_mask is correct in term of number of selected hypothesis.
                # if rejection_method == 'uniform_negative' :
                    # assert selected_confidence_mask.sum() == batch*number_unconfident+torch.sum(gt_conf_stacked_t==1), "The number of selected hypothesis is not correct."
   
                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(conf_pred_stacked[selected_confidence_mask], gt_conf_stacked_t[selected_confidence_mask])
            
            else :
                confidence_loss = torch.tensor(0.)

            sum_losses = torch.add(sum_losses, conf_weight*confidence_loss)
            
        return sum_losses