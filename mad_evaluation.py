import collections
import copy
import sys
import time
from random import seed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = r'D:\Desktop\demir\ffmpeg\bin\ffmpeg.exe'
from torch import optim

import dataset
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel, update_ema_params


import cv2
import logging
import lpips
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from skimage.metrics import structural_similarity as ssim
from torch.nn import L1Loss


def detect_anomaly(diff_model, unet_model, image, device, args_simplex):

    # returns 500 length list for every step of the diffusion process
    output = diff_model.forward_backward(
                    unet_model, image,
                    see_whole_sequence="whole",
                    t_distance=250, denoise_fn=args_simplex["noise_fn"]
                    )
    
    # take mse of output and img
    mse = (image.cpu() - output[-1]) ** 2
    # if mse is greater than threshold, it is an anomaly
    anomaly_map = mse > 0.5
    recon = output[-1].to(device)
    result = {
        'reconstruction': recon,
        'anomaly_map': anomaly_map,
        }
    
    return result


def evaluate(test_data_dict, device, criterion_rec = torch.nn.L1Loss()):
    sys.argv.append("6000")
    args_simplex, output_simplex = load_parameters(device)

    unet_simplex = UNetModel(
            args_simplex['img_size'][0], args_simplex['base_channels'], channel_mults=args_simplex['channel_mults']
            )
    
    betas = get_beta_schedule(args_simplex['T'], args_simplex['beta_schedule'])

    diff_simplex = GaussianDiffusionModel(
            args_simplex['img_size'], betas, loss_weight=args_simplex['loss_weight'],
            loss_type=args_simplex['loss-type'], noise=args_simplex["noise_fn"]
            )
    
    unet_simplex.load_state_dict(output_simplex["ema"])
    unet_simplex.eval()
    unet_simplex.to(device)

    lpips_alex = lpips.LPIPS(net='alex')  # best forward scores
    metrics = {
            'L1': [],
            'LPIPS': [],
            'SSIM': [],
            'TP': [],
            'FP': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
        }
    for dataset_key in test_data_dict.keys():
        dataset = test_data_dict[dataset_key]
        test_metrics = {
            'L1': [],
            'LPIPS': [],
            'SSIM': [],
            'TP': [],
            'FP': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
        }
        print('******************* DATASET: {} ****************'.format(dataset_key))
        tps, fns, fps = 0, 0, []
        for idx, data in enumerate(dataset):
            inputs, masks, neg_masks = data['image'].to(device), data['pos_mask'].to(device), data['neg_mask'].to(device)
            nr_batches, nr_slices, width, height = inputs.shape
            neg_masks[neg_masks > 0.5] = 1
            neg_masks[neg_masks < 1] = 0
            results = detect_anomaly(diff_simplex, unet_simplex, inputs, device, args_simplex)
            reconstructions = results['reconstruction']
            anomaly_maps = results['anomaly_map']
            
            for i in range(nr_batches):
                count = str(idx * nr_batches + i)
                x_i = inputs[i][0]
                x_rec_i = reconstructions[i][0] if reconstructions is not None else None
                ano_map_i = anomaly_maps[i][0].detach().numpy()
                mask_i = masks[i][0].cpu().detach().numpy()
                neg_mask_i = neg_masks[i][0].cpu().detach().numpy()
                bboxes = cv2.cvtColor(neg_mask_i * 255, cv2.COLOR_GRAY2RGB)
                cnts_gt = cv2.findContours((mask_i * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts_gt = cnts_gt[0] if len(cnts_gt) == 2 else cnts_gt[1]
                gt_box = []
                    
                for c_gt in cnts_gt:
                    x, y, w, h = cv2.boundingRect(c_gt)
                    gt_box.append([x, y, x + w, y + h])
                    cv2.rectangle(bboxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    
                if x_rec_i is not None:
                    loss_l1 = criterion_rec(x_rec_i, x_i)
                    test_metrics['L1'].append(loss_l1.item())
                    loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                    test_metrics['LPIPS'].append(loss_lpips)
                    x_rec_i = x_rec_i.cpu().detach().numpy()
                    ssim_ = ssim(x_rec_i, x_i.cpu().detach().numpy(), data_range=1.)
                    test_metrics['SSIM'].append(ssim_)
                    
                x_i = x_i.cpu().detach().numpy()
                x_pos = ano_map_i * mask_i
                x_neg = ano_map_i * neg_mask_i
                res_anomaly = np.sum(x_pos)
                res_healthy = np.sum(x_neg)
                
                amount_anomaly = np.count_nonzero(x_pos)
                amount_mask = np.count_nonzero(mask_i)
                
                tp = 1 if amount_anomaly > 0.1 * amount_mask else 0  # 10% overlap due to large bboxes e.g. for enlarged ventricles
                tps += tp
                fn = 1 if tp == 0 else 0
                fns += fn
                
                fp = int(res_healthy / max(res_anomaly, 1))
                fps.append(fp)
                precision = tp / max((tp + fp), 1)
                test_metrics['TP'].append(tp)
                test_metrics['FP'].append(fp)
                test_metrics['Precision'].append(precision)
                test_metrics['Recall'].append(tp)
                test_metrics['F1'].append(2 * (precision * tp) / (precision + tp + 1e-8))
                    
                if int(count) == 0:
                    if x_rec_i is None:
                        x_rec_i = np.zeros(x_i.shape)
                    elements = [x_i, x_rec_i, ano_map_i, bboxes.astype(np.int64), x_pos, x_neg]
                    v_maxs = [1, 1, 0.99, 1, np.max(ano_map_i), np.max(ano_map_i)]
                    titles = ['Input', 'Reconstruction', 'Anomaly Map', 'GT',
                              str(np.round(res_anomaly, 2)) + ', TP: ' + str(tp),
                              str(np.round(res_healthy, 2)) + ', FP: ' + str(fp)]
                    diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                    diffp.set_size_inches(len(elements) * 4, 4)
                    for idx_arr in range(len(axarr)):
                        axarr[idx_arr].axis('off')
                        v_max = v_maxs[idx_arr]
                        c_map = 'gray' if v_max == 1 else 'plasma'
                        axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)
                        axarr[idx_arr].set_title(titles[idx_arr])

        for metric in test_metrics:
            print('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                              np.nanstd(test_metrics[metric])))
            if metric == 'TP':
                print(f'TP: {np.sum(test_metrics[metric])} of {len(test_metrics[metric])} detected')
            if metric == 'FP':
                print(f'FP: {np.sum(test_metrics[metric])} missed')
            metrics[metric].append(test_metrics[metric])

    print('Writing plots...')
    fig_bps = dict()
    for metric in metrics:
        fig_bp = go.Figure()
        x = []
        y = []
        for idx, dataset_values in enumerate(metrics[metric]):
            dataset_name = list(test_data_dict)[idx]
            for dataset_val in dataset_values:
                y.append(dataset_val)
                x.append(dataset_name)

        fig_bp.add_trace(go.Box(
            y=y,
            x=x,
            name=metric,
            boxmean='sd'
        ))
        title = 'score'
        fig_bp.update_layout(
            yaxis_title=title,
            boxmode='group',  # group together boxes of the different traces for each value of x
            yaxis=dict(range=[0, 1]),
        )
        fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)
        fig_bps[metric] = fig_bp
    return metrics, fig_bps, diffp

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data_dict = dataset.get_all_test_dataloaders(
        r'C:\Users\dmrar\Desktop\WS23-24\unsupervised anomaly detection\AnoDDPM\data\splits',
        (128, 128),
        1,)

    metrics, fig_bps, diffp = evaluate(test_data_dict, device)