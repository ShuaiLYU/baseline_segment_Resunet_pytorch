
import  torch
import  numpy as np
def iter_on_a_batch(batch, model, losses, optim, phase, device):
    assert isinstance(losses, dict)
    assert phase in ["train", "valid", "test", "infer"]
    img_batch, label_pixel_batch, label_batch, file_name_batch = batch
    img_rensor = torch.tensor(img_batch).float().to(device)
    label_tensor = torch.tensor(label_batch).float().to(device)
    # forward

    mask_tensor = model(img_rensor) # 在devie
    mask_batch = mask_tensor.detach().cpu().numpy()  # cpu

    ###### cul loss
    if phase in ["train", "valid", "test"]:

        label_pixel_tensor = torch.tensor(label_pixel_batch).float().to(device)  # gpu
        loss_segment = losses["supervise"](mask_tensor.squeeze(1), label_pixel_tensor.squeeze(1))
        loss_dict = {"segment": loss_segment.mean()}
    ##### backward
    if phase in ["train"]:
        assert isinstance(loss_dict, dict)
        model.zero_grad()
        if len(loss_dict)==1:
            loss_sum=list(loss_dict.values())[0]
        else:
            loss_sum = sum(*list(loss_dict.values()))
        loss_sum.backward()
        optim.step()
        #### return
    result = {"mask_batch": mask_batch, }
    if phase in ["train", "valid", "test"]:
        for key, loss in loss_dict.items():
            loss_dict[key] = float(loss)
    result["loss"] = loss_dict
    return result


def iter_on_a_epoch(epo,phase,data_loader,model,losses,optim,metrics,writer,visualer,device):
    # train
    metrics.reset()
    epo_loss = {}
    for cnt_batch, batch in enumerate(data_loader):

        result = iter_on_a_batch(batch, model,
                                 losses=losses,
                                 optim=optim,
                                 phase=phase, device=device)

        img_batch, label_pixel_batch, _, file_name_batch = batch
        #可视化图像
        img_batch = img_batch.detach().cpu().numpy()
        label_pixel_batch = label_pixel_batch.detach().cpu().numpy()
        visual_list = [img_batch.transpose(0, 2, 3, 1) * 255,
                       label_pixel_batch.transpose(0, 2, 3, 1) * 255,
                       result["mask_batch"].transpose(0, 2, 3, 1) * 255]
        visualer.write(epo,cnt_batch,visual_list,
                       child_dir="epo-{}_{}".format(epo,phase),
                       file_name_batch=file_name_batch)

        # 每次迭代打印损失函数
        loss_dict = result["loss"]
        # s = "epoch:{},batch:{},lr:{:.4f}".format(epo, cnt_batch, float(optim.state_dict()['param_groups'][0]['lr']))
        # for key, val in loss_dict.items():
        #     s += ",{}_loss:{:.4f}".format(key, float(val))
        # print(s)
        # 添加到writer
        writer.add_scalars("step_loss", loss_dict,(epo-1)*len(data_loader)+cnt_batch)
        # 添加到epo_loss
        for key, val in loss_dict.items():
            key = key + "_loss"
            if key not in epo_loss.keys():
                epo_loss[key] = list()
            else:
                epo_loss[key].append(val)
        # 添加结果到metrcis
        metrics.addBatch(np.where(result["mask_batch"]>0.3,1,0), label_pixel_batch.astype(np.int64))

    iou_defect = metrics.clsIntersectionOverUnion(1)
    #对epoch_loss求平均
    for key, val in epo_loss.items():
        epo_loss[key] = np.array(val).sum()/ len(val)

    s="----epoch:{},{},iou:{:.4f}".format(epo,phase, iou_defect)
    for key, val in epo_loss.items():
        s+=",{}:{:.4f}".format(key,val)
    print(s)
    writer.add_scalar("iou_defect_{}".format(phase), iou_defect, global_step=epo)
    writer.add_scalars("loss_{}".format(phase), epo_loss, global_step=epo)
