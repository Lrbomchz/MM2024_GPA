import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import time


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, training_step,
                    log_writer=None,
                    args=None):
    model.train(True)
    #teacher_model.eval()
    #for name, parameter in teacher_model.named_parameters():
        #print(name, parameter.requires_grad)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        #samples_detach = samples.detach()
        
        #print("1", time.time())
        #with torch.no_grad():
            #weight = teacher_model(samples_detach)
        
            #weight = torch.tensor(teacher_model(samples_detach), requires_grad=False, device=samples.device)
        #teacher_model(samples)
        #weight = None
        #weight.grad = None
        
        #print("2", time.time())
        
        with torch.cuda.amp.autocast():
            #teacher_model(samples)
            loss, _, _, _ = model(samples, training_step, mask_ratio=args.mask_ratio)
        #print("3", time.time())
            
        # loss_value = loss.detach().float()
        #loss_value = torch.tensor(loss, requires_grad=False)
        #loss_value = loss.item()
        loss_value = loss.item()
        #print("4", time.time())
        
        #time3 = time.time()

        if not math.isfinite(loss_value):
            print(loss_value)
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        #print("5", time.time())
        #time3_5 = time.time()
        loss /= accum_iter
        #time4 = time.time()
        #time3 = time.time()
        #print("6", time.time())
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        #print("7", time.time())
            
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        #print(time4-time3_5, time3_5-time3)
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}