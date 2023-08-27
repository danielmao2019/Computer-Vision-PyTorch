import torch
import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        train_epochs,
        criterion,
        optimizer,
        metrics,
        save_interval,
        resume_from,
        device,
        logger,
    ):
        assert isinstance(model, torch.nn.Module)
        assert isinstance(train_dataloader, torch.utils.data.DataLoader)
        assert isinstance(eval_dataloader, torch.utils.data.DataLoader)
        assert type(train_epochs) == int
        assert isinstance(criterion, torch.nn.Module)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert type(metrics) == dict
        assert type(save_interval) == int
        assert type(resume_from) == str
        assert type(device) == torch.device
        self.model = model
        self.train_dataloader = train_dataloader
        self.start_epoch = None
        self.train_epochs = train_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.save_interval = save_interval
        self.device = device
        self.buffer = {
            'train_losses': [],
            'train_scores': {},
        }
        self.logger = logger
        ###
        self.model.train()
        self.model.to(self.device)
        if resume_from is not None:
            assert os.path.exists(resume_from)
            self._load_model(resume_from)
        else:
            self.start_epoch = 0

    def _train_single_epoch(self):
        for element in tqdm.tqdm(self.train_dataloader, leave=False):
            images, labels = element[0].to(self.device), element[1].to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            self.buffer['train_losses'].append(loss.item())
            ###
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            ###
            for key, val in self.metric.items():
                self.buffer['train_scores'][key].append(self.metric[key](outputs, labels))

    def _save_model(self, cur_epoch, filepath):
        """
        Args:
            cur_epoch (int): the index of the total finished epoch.
            filepath (str): the path to which the checkpoint will be saved.
        Returns:
            None
        """
        checkpoint = {
            'cur_epoch': cur_epochs + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        if os.path.exists(filepath):
            os.chmod(filepath, 0o600)
        torch.save(checkpoint, filepath)
        os.chmod(filepath, 0o400)

    def _load_model(self, filepath):
        """
        Args:
            filepath (str): the path to which the checkpoint was saved.
        Returns:
            None
        """
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        assert self.start_epoch is None
        self.start_epoch = checkpoint['cur_epoch']

    def train(self):
        assert self.start_epoch is not None
        for epoch in range(self.start_epoch, self.start_epoch+self.train_epochs):
            self._train_single_epoch()
            if epoch % self.save_interval == 0:
                self._save_model()
