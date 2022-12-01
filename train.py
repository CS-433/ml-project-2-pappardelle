from torchvision import datasets, transforms
import torch

class Trainer: 
    " Trainer class"
    
    def __init__(self, num_epochs, model, device, criterion, optimizer, train_loader,
                 valid_loader=None, lr_scheduler=None):
        
        self.num_epochs = num_epochs
        self.model = model
        self.device = device 
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.len_epoch = len(self.train_data_loader)
        self.valid_loader = valid_loader
        
        self.start_epoch = 1
        
    "training function"
    def training(self): 
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []
        
        for epoch in range(self.start_epoch, self.num_epochs+1):
            train_loss, train_acc = train_epoch(self, epoch)
            
            train_loss_history.extend(train_loss)
            train_acc_history.extend(train_acc)
            
            val_loss, val_acc = validate(self)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            
        # ===== Plot training curves =====
        n_train = len(train_acc_history)
        t_train = self.num_epochs * np.arange(n_train) / n_train
        t_val = np.arange(1, self.num_epochs + 1)
        plt.figure()
        plt.plot(t_train, train_acc_history, label="Train")
        plt.plot(t_val, val_acc_history, label="Val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.figure()
        plt.plot(t_train, train_loss_history, label="Train")
        plt.plot(t_val, val_loss_history, label="Val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        # ===== Plot low/high loss predictions on validation set =====
        points = get_predictions(self, partial(torch.nn.functional.cross_entropy, reduction="none"),
        )
        points.sort(key=lambda x: x[1])
        plt.figure(figsize=(15, 6))
        for k in range(5):
            plt.subplot(2, 5, k + 1)
            plt.imshow(points[k][0].reshape(28, 28), cmap="gray")
            plt.title(f"true={int(points[k][3])} pred={int(points[k][2])}")
            plt.subplot(2, 5, 5 + k + 1)
            plt.imshow(points[-k - 1][0].reshape(28, 28), cmap="gray")
            plt.title(f"true={int(points[-k-1][3])} pred={int(points[-k-1][2])}")
        

    def train_epoch(self, epoch):
        self.model.train() #set model to train mode 

        loss_history = []
        accuracy_history = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()

            loss_history.append(loss.item())
            accuracy_history.append(correct / len(data))

            if batch_idx % (len(self.train_loader.dataset) // len(data) // 10) == 0:
                print(
                    f"Train Epoch: {epoch}-{batch_idx} batch_loss={loss.item()/len(data):0.2e} batch_acc={correct/len(data):0.3f}"
                )


        return loss_history, accuracy_history


    @torch.no_grad()
    def validate(self): 
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            test += self.criterion(output, target).item() * len(data)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.val_loader.dataset)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss,
                correct,
                len(self.val_loader.dataset),
                100.0 * correct / len(self.val_loader.dataset),
            )
        )

        return test_loss, correct / len(self.val_loader.dataset)

    @torch.no_grad()
    def get_predictions(self, num=None):
        self.model.eval()
        points = []
        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            data = np.split(data.cpu().numpy(), len(data))
            loss = np.split(loss.cpu().numpy(), len(data))
            pred = np.split(pred.cpu().numpy(), len(data))
            target = np.split(target.cpu().numpy(), len(data))
            points.extend(zip(data, loss, pred, target))

            if num is not None and len(points) > num:
                break

        return points






