def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    for (i, pack) in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            (images, gts) = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            (P1, P2, P3, P4, P5) = model(images)
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)
            loss_P5 = structure_loss(P5, gts)
            loss = 1.2 * loss_P1 + 0.6 * loss_P2 + 0.3 * loss_P3 + 0.15 * loss_P4 + 0.1 * loss_P5
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
        if i % 20 == 0 or i == total_step:
            print('{} Epoch[{:03d}/{:03d}], Step[{:04d}/{:04d}], loss-all:[{:0.4f}],P1:[{:0.4f}] lr:[{:0.7f}]'.format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show(), loss_P1, optimizer.param_groups[0]['lr']))
            logging.info('{} Epoch[{:03d}/{:03d}], Step[{:04d}/{:04d}], loss-all:[{:0.4f}],P1:[{:0.4f}] lr:[{:0.7f}]'.format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show(), loss_P1, optimizer.param_groups[0]['lr']))
    save_path = opt.train_save
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    global dict_plot
