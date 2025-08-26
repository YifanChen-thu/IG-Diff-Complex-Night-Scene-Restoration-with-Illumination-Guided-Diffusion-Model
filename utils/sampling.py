import torch
import utils.logging
import os
import torchvision
from torchvision.transforms.functional import crop


# This script is adapted from the following repository: https://github.com/ermongroup/ddim

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps_overlapping(x, x_cond, x_illu, seq, model, b, eta=0., corners=None, p_size=None, manual_batching=True):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        x_grid_mask = torch.zeros_like(x_cond, device=x.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(1) * i).to(x.device)
            next_t = (torch.ones(1) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et_output = torch.zeros_like(x_cond, device=x.device)
            
            # manual_batching_size = 64 if p_size == 64 else 32   #snow condition  7720M  20s
            #22222
            manual_batching_size = 330   #condi  35s/700*400 mini     lolblur51s  48s    85s     88s sdsdindoor 
            
            # manual_batching_size = 420   #condi mini multi
            # manual_batching_size = 500   #condi mini single lol-blur

            
            xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
            x_cond_patch = torch.cat([crop(x_cond, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
            x_illu_patch = torch.cat([crop(x_illu, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)

            for i in range(0, len(corners), manual_batching_size):
                outputs = model.module(torch.cat([x_cond_patch[i:i+manual_batching_size], 
                                                  xt_patch[i:i+manual_batching_size]], dim=1), 
                                       t,
                                       illu=x_illu_patch[i:i+manual_batching_size])
                
                for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                    et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
           
            et = torch.div(et_output, x_grid_mask)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds