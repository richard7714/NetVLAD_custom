import torch
import torch.nn as nn


def train(model, DataLoader,
          epochs,global_batch_size, lr, momentum,weightDecay,losses, best_loss, margin):

    criterion = nn.TripletMarginLoss(margin=margin ** 0.5, p=2, reduction='sum').cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum,
                                weight_decay=weightDecay)

    model.train()

    for epoch in range(epochs):
        for batch_idx, (train_image, train_label) in enumerate(DataLoader):
            output_train = model.encoder(train_image.squeeze().cuda())
            output_train = model.pool(output_train)
            triplet_loss = criterion(output_train[0].reshape(1, -1), output_train[1].reshape(1, -1),
                                     output_train[2].reshape(1, -1))

            if batch_idx == 0:
                optimizer.zero_grad()

            triplet_loss.backward(retain_graph=True)
            losses.update(triplet_loss.item())

            if (batch_idx + 1) % global_batch_size == 0:
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        p.grad /= global_batch_size

                    optimizer.step()
                    optimizer.zero_grad()

            if batch_idx % 20 == 0 and batch_idx != 0:
                print('epoch : {}, batch_idx  : {}, triplet_loss : {}'.format(epoch, batch_idx, losses.avg))

        if best_loss > losses.avg:
            best_save_name = 'best_model.pt'
            best_path = F"./ckpt/{best_save_name}"
            torch.save(model.state_dict(), best_path)

        model_save_name = 'model_{:02d}.pt'.format(epoch)
        path = F"./ckpt/{model_save_name}"
        torch.save(model.state_dict(), path)
